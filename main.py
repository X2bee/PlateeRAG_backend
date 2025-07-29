from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
import os
from contextlib import asynccontextmanager
from controller.nodeController import router as nodeRouter
from controller.configController import router as configRouter
from controller.workflowController import router as workflowController
from controller.nodeStateController import router as nodeStateRouter
from controller.performanceController import router as performanceRouter
from controller.embeddingController import router as embeddingRouter
from controller.retrievalController import router as retrievalRouter
from controller.interactionController import router as interactionRouter
from controller.appController import router as appRouter
from controller.authController import router as authRouter
from controller.vastController import router as vastRouter
from controller.nodeApiController import router as nodeApiRouter, register_node_api_routes
from editor.node_composer import run_discovery, generate_json_spec, get_node_registry
from config.config_composer import config_composer
from service.database import AppDatabaseManager
from service.database.models import APPLICATION_MODELS
from service.retrieval import RAGService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("plateerag-backend")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 라이프사이클 관리"""
    try:
        logger.info("Starting application lifespan...")

        # 1. 데이터베이스 설정만 먼저 초기화
        database_config = config_composer.initialize_database_config_only()
        if not database_config:
            logger.error("Failed to initialize database configuration")
            return

        # 2. 애플리케이션 데이터베이스 초기화 (모든 모델 테이블 생성)
        logger.info("Initializing application database...")
        app_db = AppDatabaseManager(database_config)
        app_db.register_models(APPLICATION_MODELS)

        if app_db.initialize_database():
            app.state.app_db = app_db
            logger.info("Application database initialized successfully")

            # Run database migrations
            if app_db.run_migrations():
                logger.info("Database migrations completed successfully")
            else:
                logger.warning("Database migrations failed, but continuing startup")
        else:
            logger.error("Failed to initialize application database")
            app.state.app_db = None
            return

        # 3. 나머지 설정들 초기화 (이제 DB 테이블이 존재함)
        configs = config_composer.initialize_remaining_configs()
        app.state.config = configs
        app.state.config_composer = config_composer

        # 4. RAG 서비스 초기화 (벡터 DB와 임베딩 제공자)
        try:
            logger.info("Initializing RAG services...")
            rag_service = RAGService(configs["vectordb"], configs.get("openai"))

            # 개별 서비스들을 app.state에 등록
            app.state.rag_service = rag_service
            app.state.vector_manager = rag_service.vector_manager
            app.state.embedding_client = rag_service.embeddings_client
            app.state.document_processor = rag_service.document_processor

            logger.info("RAG services initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize RAG services: {e}")
            # RAG 서비스 초기화 실패 시에도 애플리케이션 시작은 계속
            app.state.rag_service = None
            app.state.vector_manager = None
            app.state.embedding_client = None
            app.state.document_processor = None

        config_composer.ensure_directories()
        validation_result = config_composer.validate_critical_configs()
        if not validation_result["valid"]:
            for error in validation_result["errors"]:
                logger.error(f"Configuration error: {error}")
        for warning in validation_result["warnings"]:
            logger.warning(f"Configuration warning: {warning}")

        if configs["node"].AUTO_DISCOVERY.value:
            logger.info("Starting node discovery...")
            run_discovery()
            registry_path = configs["node"].REGISTRY_FILE_PATH.value
            generate_json_spec(registry_path)
            app.state.node_registry = get_node_registry()
            app.state.node_count = len(app.state.node_registry)

            # 노드 API 라우트 등록
            logger.info("Registering node API routes...")
            register_node_api_routes()
            logger.info("Node API routes registered successfully")

            logger.info(f"Node discovery completed! Registered {app.state.node_count} nodes")
        else:
            logger.info("Node auto-discovery is disabled")

        logger.info("Application startup complete!")

    except Exception as e:
        logger.error(f"Error during startup: {e}")
        logger.info("Application will continue despite startup error")

    yield

    logger.info("Application shutdown...")
    try:
        # 애플리케이션 데이터베이스 정리
        if hasattr(app.state, 'app_db') and app.state.app_db:
            app.state.app_db.close()
            logger.info("Application database connection closed")

        config_composer.save_all()
        logger.info("Configurations saved on shutdown")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

app = FastAPI(
    title="PlateeRAG Backend",
    description="API for training models with customizable parameters - Enhanced with app.state",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 시작시 기본값, config 로드 후 업데이트 가능
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(nodeRouter)
app.include_router(authRouter)
app.include_router(configRouter)
app.include_router(workflowController)
app.include_router(nodeStateRouter)
app.include_router(performanceRouter)
app.include_router(embeddingRouter)
app.include_router(retrievalRouter)
app.include_router(appRouter)
app.include_router(vastRouter)
app.include_router(nodeApiRouter)

# 기존 /app 엔드포인트들은 appController로 이동했으므로 여기서 제거

if __name__ == "__main__":
    try:
        host = os.environ.get("APP_HOST", "0.0.0.0")
        port = int(os.environ.get("APP_PORT", "8000"))
        debug = os.environ.get("DEBUG_MODE", "false").lower() in ('true', '1', 'yes', 'on')

        print(f"Starting server on {host}:{port} (debug={debug})")
        uvicorn.run("main:app", host=host, port=port, reload=debug)
    except Exception as e:
        logger.warning(f"Failed to load config for uvicorn: {e}")
        logger.info("Using default values for uvicorn")
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

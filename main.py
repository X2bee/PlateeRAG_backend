from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
import os
from contextlib import asynccontextmanager
from controller.node.nodeApiController import register_node_api_routes

from controller.node.router import node_router
from controller.admin.router import admin_router
from controller.manager.router import manager_router
from controller.workflow.router import workflow_router
from controller.rag.router import rag_router
from controller.audio.router import audio_router

from controller.trainController import router as trainRouter
from controller.llmController import router as llmRouter
from controller.performanceController import router as performanceRouter
from controller.interactionController import router as interactionRouter
from controller.huggingface.huggingfaceController import router as huggingfaceRouter
from controller.appController import router as appRouter
from controller.authController import router as authRouter
from controller.vastController import router as vastRouter
from editor.node_composer import run_discovery, generate_json_spec, get_node_registry
from editor.async_workflow_executor import execution_manager
from config.config_composer import config_composer
from service.database.models import APPLICATION_MODELS

from service.database import AppDatabaseManager
from service.embedding.embedding_factory import EmbeddingFactory
from service.stt.stt_factory import STTFactory
from service.tts.tts_factory import TTSFactory
from service.vast.vast_service import VastService
from service.vector_db.vector_manager import VectorManager
from service.retrieval.document_processor.document_processor import DocumentProcessor
from service.retrieval.document_info_generator.document_info_generator import DocumentInfoGenerator
from service.sync.workflow_deploy_sync import sync_workflow_deploy_meta
from controller.workflow.utils.workflow_helpers import workflow_data_synchronizer

def print_xgen_logo():
    logo = """
    ██╗  ██╗ ██████╗ ███████╗███╗   ██╗
    ╚██╗██╔╝██╔════╝ ██╔════╝████╗  ██║
     ╚███╔╝ ██║  ███╗█████╗  ██╔██╗ ██║
     ██╔██╗ ██║   ██║██╔══╝  ██║╚██╗██║
    ██╔╝ ██╗╚██████╔╝███████╗██║ ╚████║
    ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝

    🚀 PlateeRAG Backend with XGEN Engine 🚀
    """
    print(logo)

def print_step_banner(step_num, title, description=""):
    """각 단계별 배너 출력"""
    banner = f"""
    ┌{'─' * 60}┐
    │  STEP {step_num}: {title:<50}│
    {f'│  {description:<58}│' if description else ''}
    └{'─' * 60}┘
    """
    print(banner)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 라이프사이클 관리"""
    try:
        print_xgen_logo()
        logger.info("🌟 Starting XGEN application lifespan...")

        # 1. 데이터베이스 설정만 먼저 초기화
        print_step_banner(1, "DATABASE INITIALIZATION", "Initializing core database configuration")
        logger.info("⚙️  Step 1: Database configuration initialization starting...")

        database_config = config_composer.initialize_database_config_only()
        if not database_config:
            logger.error("❌ Failed to initialize database configuration")
            return
        logger.info("✅ Step 1: Database configuration initialized successfully!")

        # 2. 애플리케이션 데이터베이스 초기화 (모든 모델 테이블 생성)
        print_step_banner(2, "APPLICATION DATABASE SETUP", "Creating tables and running migrations")
        logger.info("⚙️  Step 2: Application database initialization starting...")

        app_db = AppDatabaseManager(database_config)
        app_db.register_models(APPLICATION_MODELS)

        if app_db.initialize_database():
            app.state.app_db = app_db
            logger.info("✅ Step 2: Application database initialized successfully!")

            # Run database migrations
            logger.info("🔄 Running database migrations...")
            if app_db.run_migrations():
                logger.info("✅ Database migrations completed successfully!")
            else:
                logger.warning("⚠️  Database migrations failed, but continuing startup")
        else:
            logger.error("❌ Failed to initialize application database")
            app.state.app_db = None
            return

        # 3. 나머지 설정들 초기화 (이제 DB 테이블이 존재함)
        print_step_banner(3, "CONFIGURATION SETUP", "Loading remaining system configurations")
        logger.info("⚙️  Step 3: System configuration initialization starting...")

        configs = config_composer.initialize_remaining_configs()
        app.state.config_composer = config_composer
        logger.info("✅ Step 3: System configurations loaded successfully!")

        # 4. RAG 서비스 초기화 (벡터 DB와 임베딩 제공자)
        print_step_banner(4, "RAG SERVICES INITIALIZATION", "Setting up vector DB and embedding services")
        try:
            print_step_banner(4.1, "EMBEDDING SERVICE SETUP", "Setting up embedding services")
            embedding_client = EmbeddingFactory.create_embedding_client(config_composer)
            app.state.embedding_client = embedding_client

            print_step_banner(4.2, "VECTOR SERVICE SETUP", "Setting up vector services")
            vector_manager = VectorManager(config_composer)
            app.state.vector_manager = vector_manager

            print_step_banner(4.3, "DOCUMENT PROCESSOR SETUP", "Setting up document processing services")
            document_processor = DocumentProcessor(config_composer)
            app.state.document_processor = document_processor

            print_step_banner(4.4, "DOCUMENT INFO GENERATOR SETUP", "Setting up document info generation services")
            document_info_generator = DocumentInfoGenerator(config_composer)
            app.state.document_info_generator = document_info_generator
            logger.info("✅ Step 4: RAG services components initialized successfully!")

        except Exception as e:
            logger.error(f"❌ Step 4: Failed to initialize RAG services: {e}")
            # RAG 서비스 초기화 실패 시에도 애플리케이션 시작은 계속
            app.state.rag_service = None
            app.state.vector_manager = None
            app.state.embedding_client = None
            app.state.document_processor = None

        # 5. STT 서비스 초기화
        if config_composer.get_config_by_name("IS_AVAILABLE_STT").value:
            print_step_banner(5, "STT SERVICE SETUP", "Setting up Speech-to-Text services")
            try:
                logger.info("⚙️  Step 5: STT service initialization starting...")
                stt_client = STTFactory.create_stt_client(config_composer)
                app.state.stt_service = stt_client
                logger.info("✅ Step 5: STT service initialized successfully!")
            except Exception as e:
                logger.error(f"❌ Step 5: Failed to initialize STT service: {e}")
                # STT 서비스 초기화 실패 시에도 애플리케이션 시작은 계속
                app.state.stt_service = None
        else:
            print_step_banner(5, "STT SERVICE SETUP", "STT service is disabled in configuration")
            app.state.stt_service = None

        # 5.5. TTS 서비스 초기화
        if config_composer.get_config_by_name("IS_AVAILABLE_TTS").value:
            print_step_banner(5.5, "TTS SERVICE SETUP", "Setting up Text-to-Speech services")
            try:
                logger.info("⚙️  Step 5.5: TTS service initialization starting...")
                tts_client = TTSFactory.create_tts_client(config_composer)
                app.state.tts_service = tts_client
                logger.info("✅ Step 5.5: TTS service initialized successfully!")
            except Exception as e:
                logger.error(f"❌ Step 5.5: Failed to initialize TTS service: {e}")
                # TTS 서비스 초기화 실패 시에도 애플리케이션 시작은 계속
                app.state.tts_service = None
        else:
            print_step_banner(5.5, "TTS SERVICE SETUP", "TTS service is disabled in configuration")
            app.state.tts_service = None

        # 6. vast_service Instance 생성
        print_step_banner(6, "VAST SERVICE SETUP", "Initializing cloud compute management")
        logger.info("⚙️  Step 6: VAST service initialization starting...")
        app.state.vast_service = VastService(app_db, config_composer)
        logger.info("✅ Step 6: VAST service initialized successfully!")

        # 7. 워크플로우 실행 매니저 초기화
        print_step_banner(7, "WORKFLOW MANAGER SETUP", "Setting up workflow execution engine")
        logger.info("⚙️  Step 7: Workflow execution manager initialization starting...")
        app.state.execution_manager = execution_manager
        logger.info("✅ Step 7: Workflow execution manager initialized successfully!")

        print_step_banner(8, "SYSTEM VALIDATION", "Validating configurations and directories")
        logger.info("⚙️  Step 8: System validation starting...")

        config_composer.ensure_directories()
        validation_result = config_composer.validate_critical_configs()
        if not validation_result["valid"]:
            for error in validation_result["errors"]:
                logger.error(f"❌ Configuration error: {error}")
        for warning in validation_result["warnings"]:
            logger.warning(f"⚠️  Configuration warning: {warning}")
        logger.info("✅ Step 8: System validation completed!")

        # 9. 워크플로우 데이터 동기화
        print_step_banner(9, "WORKFLOW DATA SYNC", "Synchronizing workflow filesystem and database")
        logger.info("⚙️  Step 9: Workflow data synchronization starting...")

        try:
            sync_result = await workflow_data_synchronizer(app.state.app_db)
            if sync_result["success"]:
                logger.info(f"✅ Step 9: Workflow data sync completed successfully! "
                           f"Added {sync_result['files_added_to_db']} to DB, "
                           f"Created {sync_result['files_created_from_db']} files, "
                           f"Removed {sync_result['orphaned_db_entries_removed']} orphaned entries, "
                           f"Processed {sync_result['users_processed']} users")
            else:
                logger.warning(f"⚠️  Step 9: Workflow data sync completed with issues. "
                             f"Errors: {sync_result.get('errors', [])}")
        except Exception as e:
            logger.error(f"❌ Step 9: Failed to sync workflow data: {e}")

        # 9.5. 워크플로우-배포 메타데이터 동기화
        print_step_banner(9.5, "WORKFLOW-DEPLOY SYNC", "Synchronizing workflow and deploy metadata")
        logger.info("⚙️  Step 9.5: Workflow-deploy metadata synchronization starting...")

        try:
            sync_result = sync_workflow_deploy_meta(app.state.app_db)
            if sync_result["success"]:
                logger.info(f"✅ Step 9.5: Workflow-deploy sync completed successfully! "
                           f"Created {sync_result['created_deploys']} new deploy entries from "
                           f"{sync_result['total_workflows']} total workflows")
            else:
                logger.warning(f"⚠️  Step 9.5: Workflow-deploy sync completed with issues. "
                             f"Errors: {sync_result.get('errors', [])}")
        except Exception as e:
            logger.error(f"❌ Step 9.5: Failed to sync workflow-deploy metadata: {e}")

        print_step_banner(10, "NODE DISCOVERY", "Discovering and registering XGEN nodes")
        logger.info("⚙️  Step 10: Node discovery starting...")

        run_discovery()
        registry_path = configs["node"].REGISTRY_FILE_PATH.value
        generate_json_spec(registry_path)
        app.state.node_registry = get_node_registry()
        app.state.node_count = len(app.state.node_registry)

        # 노드 API 라우트 등록
        logger.info("🔗 Registering node API routes...")
        register_node_api_routes()
        logger.info("✅ Node API routes registered successfully!")

        logger.info(f"✅ Step 10: Node discovery completed! Registered {app.state.node_count} nodes")


        print_step_banner("FINAL", "XGEN STARTUP COMPLETE", "All systems operational! 🎉")
        logger.info("🎉 XGEN application startup complete! Ready to serve requests.")

    except Exception as e:
        logger.error(f"💥 Error during startup: {e}")
        logger.info("🔄 Application will continue despite startup error")

    yield

    logger.info("🛑 XGEN application shutdown...")
    print("""
    ┌─────────────────────────────────────────────────────┐
    │                XGEN SHUTDOWN                        │
    │           Gracefully stopping all services         │
    └─────────────────────────────────────────────────────┘
    """)
    try:
        # 워크플로우 실행 매니저 정리
        if hasattr(app.state, 'execution_manager') and app.state.execution_manager:
            logger.info("🔄 Shutting down workflow execution manager...")
            app.state.execution_manager.shutdown()
            logger.info("✅ Workflow execution manager shutdown complete")

        # STT 서비스 정리
        if hasattr(app.state, 'stt_service') and app.state.stt_service:
            logger.info("🔄 Cleaning up STT service...")
            await app.state.stt_service.cleanup()
            logger.info("✅ STT service cleanup complete")

        # TTS 서비스 정리
        if hasattr(app.state, 'tts_service') and app.state.tts_service:
            logger.info("🔄 Cleaning up TTS service...")
            await app.state.tts_service.cleanup()
            logger.info("✅ TTS service cleanup complete")

        # 애플리케이션 데이터베이스 정리
        if hasattr(app.state, 'app_db') and app.state.app_db:
            app.state.app_db.close()
            logger.info("✅ Application database connection closed")

        config_composer.save_all()
        logger.info("✅ Configurations saved on shutdown")
        logger.info("👋 XGEN shutdown complete. Goodbye!")
    except Exception as e:
        logger.error(f"💥 Error during shutdown: {e}")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("plateerag-backend")

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

app.include_router(node_router)
app.include_router(admin_router)
app.include_router(manager_router)
app.include_router(workflow_router)
app.include_router(rag_router)
app.include_router(audio_router)

app.include_router(authRouter)
app.include_router(llmRouter)
app.include_router(performanceRouter)
app.include_router(trainRouter)
app.include_router(interactionRouter)
app.include_router(appRouter)
app.include_router(vastRouter)
app.include_router(huggingfaceRouter)

if __name__ == "__main__":
    try:
        host = os.environ.get("APP_HOST", "0.0.0.0")
        port = int(os.environ.get("APP_PORT", "8000"))
        debug = os.environ.get("DEBUG_MODE", "false").lower() in ('true', '1', 'yes', 'on')

        print(f"Starting server on {host}:{port} (debug={debug})")
        if debug:
            uvicorn.run("main:app", host=host, port=port, reload=True)
        else:
            uvicorn.run("main:app", host=host, port=port, reload=False)
    except Exception as e:
        logger.warning(f"Failed to load config for uvicorn: {e}")
        logger.info("Using default values for uvicorn")
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
import os
from contextlib import asynccontextmanager
from typing import Dict
from controller.nodeController import router as nodeRouter
from controller.configController import router as configRouter
from controller.workflowController import router as workflowRouter
from controller.nodeStateController import router as nodeStateRouter
from controller.performanceController import router as performanceRouter
from controller.ragController import router as ragRouter
from controller.interactionController import router as interactionRouter
from controller.chatController import router as chatRouter
from src.node_composer import run_discovery, generate_json_spec, get_node_registry
from config.config_composer import config_composer
from database import AppDatabaseManager
from models import APPLICATION_MODELS
from models.user import User
from models.chat import ChatSession
from models.performance import WorkflowExecution

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
app.include_router(configRouter)
app.include_router(workflowRouter)
app.include_router(nodeStateRouter)
app.include_router(performanceRouter)
app.include_router(ragRouter)
app.include_router(interactionRouter)
app.include_router(chatRouter)

@app.get("/app/status")
async def get_app_status():
    """애플리케이션 상태 정보 반환"""
    return {
        "config": {
            "app_name": "PlateeRAG Backend",
            "version": "1.0.0",
            "environment": app.state.config["app"].ENVIRONMENT.value,
            "debug_mode": app.state.config["app"].DEBUG_MODE.value
        },
        "node_count": getattr(app.state, 'node_count', 0),
        "available_nodes": [node["id"] for node in getattr(app.state, 'node_registry', [])],
        "status": "running"
    }

@app.get("/app/config")
async def get_app_config():
    """애플리케이션 설정 반환"""
    try:
        config_summary = config_composer.get_config_summary()
        return config_summary
    except Exception as e:
        logger.error(f"Error getting app config: {e}")
        return {"error": "Failed to get configuration"}

@app.get("/app/config/persistent")
async def get_persistent_configs():
    """모든 PersistentConfig 설정 정보 반환"""
    return config_composer.get_config_summary()

@app.put("/app/config/persistent/{config_name}")
async def update_persistent_config(config_name: str, new_value: dict):
    """특정 PersistentConfig 값 업데이트"""
    try:
        config_obj = config_composer.get_config_by_name(config_name)
        old_value = config_obj.value
        
        # 값 타입에 따라 적절히 변환
        value = new_value.get("value")
        if isinstance(config_obj.env_value, bool):
            config_obj.value = bool(value)
        elif isinstance(config_obj.env_value, int):
            config_obj.value = int(value)
        elif isinstance(config_obj.env_value, float):
            config_obj.value = float(value)
        else:
            config_obj.value = str(value)
        
        config_obj.save()
        
        return {
            "message": f"Config '{config_name}' updated successfully",
            "old_value": old_value,
            "new_value": config_obj.value
        }
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Config '{config_name}' not found")
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid value type: {e}")

@app.post("/app/config/persistent/refresh")
async def refresh_persistent_configs():
    """모든 PersistentConfig를 데이터베이스에서 다시 로드"""
    config_composer.refresh_all()
    return {"message": "All persistent configs refreshed successfully from database"}

@app.post("/app/config/persistent/save")
async def save_persistent_configs():
    """모든 PersistentConfig를 데이터베이스에 저장"""
    config_composer.save_all()
    return {"message": "All persistent configs saved successfully to database"}

@app.put("/app/config")
async def update_app_config(new_config: dict):
    """애플리케이션 설정 업데이트"""
    return {"message": "Config update not implemented yet", "received": new_config}

@app.get("/demo/users")
async def get_demo_users():
    """데모용: 사용자 목록 조회"""
    if not hasattr(app.state, 'app_db') or not app.state.app_db:
        raise HTTPException(status_code=500, detail="Application database not available")
    
    users = app.state.app_db.find_all(User, limit=10)
    return {
        "users": [user.to_dict() for user in users],
        "total": len(users)
    }

@app.post("/demo/users")
async def create_demo_user(user_data: dict):
    """데모용: 새 사용자 생성"""
    if not hasattr(app.state, 'app_db') or not app.state.app_db:
        raise HTTPException(status_code=500, detail="Application database not available")
    
    user = User(
        username=user_data.get("username", ""),
        email=user_data.get("email", ""),
        full_name=user_data.get("full_name"),
        password_hash="demo_hash_" + user_data.get("username", "")
    )
    
    user_id = app.state.app_db.insert(user)
    
    if user_id:
        user.id = user_id
        return {"message": "User created successfully", "user": user.to_dict()}
    else:
        raise HTTPException(status_code=500, detail="Failed to create user")


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
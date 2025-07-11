from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
import os
from contextlib import asynccontextmanager
from controller.nodeController import router as nodeRouter
from controller.configController import router as configRouter
from controller.workflowController import router as workflowRouter
from controller.nodeStateController import router as nodeStateRouter
from src.node_composer import run_discovery, generate_json_spec, get_node_registry
from config.config_composer import config_composer

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
        configs = config_composer.initialize_all_configs()
        app.state.config = configs
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
    return {"message": "All persistent configs refreshed successfully"}

@app.post("/app/config/persistent/save")
async def save_persistent_configs():
    """모든 PersistentConfig를 데이터베이스에 저장"""
    config_composer.save_all()
    return {"message": "All persistent configs saved successfully"}

@app.put("/app/config")
async def update_app_config(new_config: dict):
    """애플리케이션 설정 업데이트"""
    return {"message": "Config update not implemented yet", "received": new_config}


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
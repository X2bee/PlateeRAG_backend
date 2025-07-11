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
    # 시작 시 초기화
    try:
        logger.info("Starting application lifespan...")
        
        # 1. 설정 시스템 초기화
        configs = config_composer.initialize_all_configs()
        
        # 2. app.state에 설정 저장
        app.state.config = configs
        
        # 3. 필요한 디렉토리들 생성
        config_composer.ensure_directories()
        
        # 4. 설정 유효성 검증
        validation_result = config_composer.validate_critical_configs()
        if not validation_result["valid"]:
            for error in validation_result["errors"]:
                logger.error(f"Configuration error: {error}")
        for warning in validation_result["warnings"]:
            logger.warning(f"Configuration warning: {warning}")
        
        # 5. 노드 discovery 실행 (설정에 따라)
        if configs["node"].AUTO_DISCOVERY.value:
            logger.info("Starting node discovery...")
            run_discovery()
            
            # 노드 레지스트리 파일 생성
            registry_path = configs["node"].REGISTRY_FILE_PATH.value
            generate_json_spec(registry_path)
            
            # app.state에 노드 정보 저장
            app.state.node_registry = get_node_registry()
            app.state.node_count = len(app.state.node_registry)
            
            logger.info(f"Node discovery completed! Registered {app.state.node_count} nodes")
        else:
            logger.info("Node auto-discovery is disabled")
        
        logger.info("Application startup complete!")
        
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        logger.info("Application will continue despite startup error")
    
    yield  # 애플리케이션 실행
    
    # 종료 시 정리
    logger.info("Application shutdown...")
    try:
        # 설정을 데이터베이스에 저장
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

# CORS 설정은 startup 이벤트에서 추가됩니다 (lifespan에서 처리)
# 기본적으로는 모든 오리진 허용
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

# app.state를 활용하는 새로운 엔드포인트들
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
        # 설정 요약 정보 가져오기
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
    # 이 기능은 설정에 따라 구현 필요
    return {"message": "Config update not implemented yet", "received": new_config}

if __name__ == "__main__":
    # 개발 모드에서는 환경변수를 우선으로 하여 실행
    try:
        # 환경변수에서 직접 설정 읽기 (PersistentConfig보다 우선)
        host = os.environ.get("APP_HOST", "0.0.0.0")
        port = int(os.environ.get("APP_PORT", "8000"))
        debug = os.environ.get("DEBUG_MODE", "false").lower() in ('true', '1', 'yes', 'on')
        
        print(f"Starting server on {host}:{port} (debug={debug})")
        uvicorn.run("main:app", host=host, port=port, reload=debug)
    except Exception as e:
        logger.warning(f"Failed to load config for uvicorn: {e}")
        logger.info("Using default values for uvicorn")
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
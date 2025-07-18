"""
App 컨트롤러

애플리케이션 관리 API 엔드포인트를 제공합니다.
애플리케이션 상태, 설정 관리, 데모 기능 등을 담당합니다.
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Dict, Any
import logging

from service.database.models.user import User

logger = logging.getLogger("app-controller")
router = APIRouter(prefix="/app", tags=["app"])

def get_config_composer(request: Request):
    """request.app.state에서 config_composer 가져오기"""
    if hasattr(request.app.state, 'config_composer') and request.app.state.config_composer:
        return request.app.state.config_composer
    else:
        # app.state에 없으면 임시로 import해서 사용
        from config.config_composer import config_composer
        return config_composer

class ConfigUpdateRequest(BaseModel):
    value: Any

class UserCreateRequest(BaseModel):
    username: str
    email: str
    full_name: str = None

@router.get("/status")
async def get_app_status(request: Request):
    """애플리케이션 상태 정보 반환"""
    return {
        "config": {
            "app_name": "PlateeRAG Backend",
            "version": "1.0.0",
            "environment": request.app.state.config["app"].ENVIRONMENT.value,
            "debug_mode": request.app.state.config["app"].DEBUG_MODE.value
        },
        "node_count": getattr(request.app.state, 'node_count', 0),
        "available_nodes": [node["id"] for node in getattr(request.app.state, 'node_registry', [])],
        "status": "running"
    }

@router.get("/config")
async def get_app_config(request: Request):
    """애플리케이션 설정 반환"""
    try:
        config_composer = get_config_composer(request)
        config_summary = config_composer.get_config_summary()
        return config_summary
    except Exception as e:
        logger.error("Error getting app config: %s", e)
        return {"error": "Failed to get configuration"}

@router.get("/config/persistent")
async def get_persistent_configs(request: Request):
    """모든 PersistentConfig 설정 정보 반환"""
    config_composer = get_config_composer(request)
    return config_composer.get_config_summary()

@router.put("/config/persistent/{config_name}")
async def update_persistent_config(config_name: str, new_value: ConfigUpdateRequest, request: Request):
    """특정 PersistentConfig 값 업데이트"""
    try:
        config_composer = get_config_composer(request)
        config_obj = config_composer.get_config_by_name(config_name)
        old_value = config_obj.value
                
        # 값 타입에 따라 적절히 변환
        value = new_value.value
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

@router.post("/config/persistent/refresh")
async def refresh_persistent_configs(request: Request):
    """모든 PersistentConfig를 데이터베이스에서 다시 로드"""
    config_composer = get_config_composer(request)
    config_composer.refresh_all()
    return {"message": "All persistent configs refreshed successfully from database"}

@router.post("/config/persistent/save")
async def save_persistent_configs(request: Request):
    """모든 PersistentConfig를 데이터베이스에 저장"""
    config_composer = get_config_composer(request)
    config_composer.save_all()
    return {"message": "All persistent configs saved successfully to database"}

@router.put("/config")
async def update_app_config(new_config: dict):
    """애플리케이션 설정 업데이트"""
    return {"message": "Config update not implemented yet", "received": new_config}

@router.get("/demo/users")
async def get_demo_users(request: Request):
    """데모용: 사용자 목록 조회"""
    if not hasattr(request.app.state, 'app_db') or not request.app.state.app_db:
        raise HTTPException(status_code=500, detail="Application database not available")
    
    users = request.app.state.app_db.find_all(User, limit=10)
    return {
        "users": [user.to_dict() for user in users],
        "total": len(users)
    }

@router.post("/demo/users")
async def create_demo_user(request: Request, user_data: UserCreateRequest):
    """데모용: 새 사용자 생성"""
    if not hasattr(request.app.state, 'app_db') or not request.app.state.app_db:
        raise HTTPException(status_code=500, detail="Application database not available")
    
    user = User(
        username=user_data.username,
        email=user_data.email,
        full_name=user_data.full_name,
        password_hash="demo_hash_" + user_data.username
    )
    
    user_id = request.app.state.app_db.insert(user)
    
    if user_id:
        user.id = user_id
        return {"message": "User created successfully", "user": user.to_dict()}
    else:
        raise HTTPException(status_code=500, detail="Failed to create user")

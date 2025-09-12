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
from service.retrieval import RAGService
from controller.helper.singletonHelper import get_config_composer, get_vector_manager, get_rag_service, get_document_processor, get_db_manager
from controller.helper.controllerHelper import extract_user_id_from_request
from service.database.logger_helper import create_logger

logger = logging.getLogger("app-controller")
router = APIRouter(prefix="/app", tags=["app"])

class ConfigUpdateRequest(BaseModel):
    value: Any

class UserCreateRequest(BaseModel):
    username: str
    email: str
    full_name: str = None

@router.get("/status")
async def get_app_status(request: Request):
    user_id = extract_user_id_from_request(request)

    # user_id가 없는 경우에도 시스템 상태 조회는 허용하되, 로깅은 제한적으로 수행
    if user_id:
        app_db = get_db_manager(request)
        backend_log = create_logger(app_db, user_id, request)
        backend_log.info("Retrieving application status")

    config_composer = get_config_composer(request)

    try:
        node_count = getattr(request.app.state, 'node_count', 0)
        node_registry = getattr(request.app.state, 'node_registry', [])
        available_nodes = [node["id"] for node in node_registry]

        status_data = {
            "config": {
                "app_name": "PlateeRAG Backend",
                "version": "1.0.0",
                "environment": config_composer.get_config_by_name("ENVIRONMENT").value,
                "debug_mode": config_composer.get_config_by_name("DEBUG_MODE").value
            },
            "node_count": node_count,
            "available_nodes": available_nodes,
            "status": "running"
        }

        if user_id:
            backend_log.success("Successfully retrieved application status",
                              metadata={"node_count": node_count,
                                      "available_nodes_count": len(available_nodes),
                                      "environment": config_composer.get_config_by_name("ENVIRONMENT").value,
                                      "debug_mode": config_composer.get_config_by_name("DEBUG_MODE").value})

        return status_data

    except Exception as e:
        if user_id:
            backend_log.error("Failed to retrieve application status", exception=e)
        logger.error("Error getting app status: %s", e)
        raise HTTPException(status_code=500, detail="Failed to retrieve application status")

@router.get("/config")
async def get_app_config(request: Request):
    """애플리케이션 설정 반환"""
    user_id = extract_user_id_from_request(request)

    if user_id:
        app_db = get_db_manager(request)
        backend_log = create_logger(app_db, user_id, request)

    try:
        if user_id:
            backend_log.info("Retrieving application configuration")

        config_composer = get_config_composer(request)
        config_summary = config_composer.get_config_summary()

        if user_id:
            config_count = len(config_summary) if isinstance(config_summary, dict) else 0
            backend_log.success("Successfully retrieved application configuration",
                              metadata={"config_count": config_count})

        return config_summary
    except Exception as e:
        if user_id:
            backend_log.error("Failed to retrieve application configuration", exception=e)
        logger.error("Error getting app config: %s", e)
        return {"error": "Failed to get configuration"}

@router.get("/config/persistent")
async def get_persistent_configs(request: Request):
    """모든 PersistentConfig 설정 정보 반환"""
    user_id = extract_user_id_from_request(request)

    if user_id:
        app_db = get_db_manager(request)
        backend_log = create_logger(app_db, user_id, request)
        backend_log.info("Retrieving persistent configurations")

    try:
        config_composer = get_config_composer(request)
        config_summary = config_composer.get_config_summary()

        if user_id:
            config_count = len(config_summary) if isinstance(config_summary, dict) else 0
            backend_log.success("Successfully retrieved persistent configurations",
                              metadata={"config_count": config_count})

        return config_summary
    except Exception as e:
        if user_id:
            backend_log.error("Failed to retrieve persistent configurations", exception=e)
        logger.error("Error getting persistent configs: %s", e)
        raise HTTPException(status_code=500, detail="Failed to retrieve persistent configurations")

@router.put("/config/persistent/{config_name}")
async def update_persistent_config(config_name: str, new_value: ConfigUpdateRequest, request: Request):
    """특정 PersistentConfig 값 업데이트"""
    user_id = extract_user_id_from_request(request)

    if user_id:
        app_db = get_db_manager(request)
        backend_log = create_logger(app_db, user_id, request)

    try:
        if user_id:
            backend_log.info("Starting persistent config update",
                           metadata={"config_name": config_name,
                                   "new_value_type": type(new_value.value).__name__})

        config_composer = get_config_composer(request)
        update_result = config_composer.update_config(config_name, new_value.value)
        old_value = update_result["old_value"]
        new_config_value = update_result["new_value"]

        logger.info("Successfully updated config '%s': %s -> %s", config_name, old_value, new_config_value)

        response_data = {
            "message": f"Config '{config_name}' updated successfully",
            "old_value": old_value,
            "new_value": new_config_value,
            "updated_in_memory": True,
        }

        if user_id:
            backend_log.success("Successfully updated persistent configuration",
                              metadata={"config_name": config_name,
                                      "old_value": str(old_value)[:100],  # 첫 100자만 로깅
                                      "new_value": str(new_config_value)[:100],  # 첫 100자만 로깅
                                      "value_type": type(new_config_value).__name__})

        return response_data

    except KeyError as exc:
        if user_id:
            backend_log.error("Configuration not found",
                            metadata={"config_name": config_name})
        raise HTTPException(status_code=404, detail=f"Config '{config_name}' not found") from exc
    except (ValueError, TypeError) as e:
        if user_id:
            backend_log.error("Invalid configuration value", exception=e,
                            metadata={"config_name": config_name,
                                    "attempted_value": str(new_value.value)[:100],
                                    "value_type": type(new_value.value).__name__})
        raise HTTPException(status_code=400, detail=f"Invalid value type: {e}") from e
    except Exception as e:
        if user_id:
            backend_log.error("Failed to update persistent configuration", exception=e,
                            metadata={"config_name": config_name})
        logger.error("Error updating config '%s': %s", config_name, e)
        raise HTTPException(status_code=500, detail="Failed to update configuration")

@router.post("/config/persistent/refresh")
async def refresh_persistent_configs(request: Request):
    """모든 PersistentConfig를 데이터베이스에서 다시 로드"""
    user_id = extract_user_id_from_request(request)

    if user_id:
        app_db = get_db_manager(request)
        backend_log = create_logger(app_db, user_id, request)

    try:
        if user_id:
            backend_log.info("Starting persistent configurations refresh")

        config_composer = get_config_composer(request)
        config_composer.refresh_all()

        response_data = {"message": "All persistent configs refreshed successfully from database"}

        if user_id:
            backend_log.success("Successfully refreshed all persistent configurations")

        return response_data

    except Exception as e:
        if user_id:
            backend_log.error("Failed to refresh persistent configurations", exception=e)
        logger.error("Error refreshing persistent configs: %s", e)
        raise HTTPException(status_code=500, detail="Failed to refresh persistent configurations")

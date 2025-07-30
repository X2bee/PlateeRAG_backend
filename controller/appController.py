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

logger = logging.getLogger("app-controller")
router = APIRouter(prefix="/app", tags=["app"])

def get_rag_service(request: Request) -> RAGService:
    """RAG 서비스 의존성 주입"""
    config_composer = request.app.state.config_composer
    if config_composer:
        vectordb_config = config_composer.get_config_by_category_name("vectordb")
        openai_config = config_composer.get_config_by_category_name("openai")
        collection_config = config_composer.get_config_by_category_name("collection")

        return RAGService(vectordb_config, collection_config, openai_config)
    else:
        raise HTTPException(status_code=500, detail="Configuration not available")

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

        # 1. DB에 저장
        config_obj.save()

        # 2. app.state의 config도 업데이트 (메모리에서 실행 중인 설정들 동기화)
        if hasattr(request.app.state, 'config') and request.app.state.config:
            # config 카테고리별로 업데이트된 값 반영
            for category_name, category_config in request.app.state.config.items():
                if category_name != "all_configs" and hasattr(category_config, config_name):
                    # 해당 카테고리의 설정 객체도 같은 값으로 업데이트
                    setattr(category_config, config_name, config_obj)
                    logger.info("Updated app.state config for category '%s': %s = %s",
                              category_name, config_name, config_obj.value)

            # all_configs도 업데이트
            if "all_configs" in request.app.state.config:
                request.app.state.config["all_configs"][config_name] = config_obj
                logger.info("Updated app.state.all_configs: %s = %s", config_name, config_obj.value)

        # 3. config_composer의 all_configs도 업데이트 (이미 참조로 연결되어 있지만 명시적으로)
        config_composer.all_configs[config_name] = config_obj

        # 4. 관련 서비스들에게 설정 변경 알림 (필요시 재초기화)
        services_refreshed = []
        try:
            # RAG 관련 설정이 변경된 경우 RAG 서비스 갱신
            if any(keyword in config_name.lower() for keyword in ['qdrant', 'vector', 'embedding', 'openai']):
                if hasattr(request.app.state, 'rag_service') and request.app.state.rag_service:
                    # RAG 서비스의 설정 참조 갱신
                    if 'vectordb' in request.app.state.config:
                        request.app.state.rag_service.config = request.app.state.config['vectordb']
                    if 'openai' in request.app.state.config:
                        request.app.state.rag_service.openai_config = request.app.state.config['openai']
                    services_refreshed.append("rag_service")
                    logger.info("Refreshed RAG service configuration")

                # 벡터 매니저 설정 갱신
                if hasattr(request.app.state, 'vector_manager') and request.app.state.vector_manager:
                    if 'vectordb' in request.app.state.config:
                        request.app.state.vector_manager.config = request.app.state.config['vectordb']
                    services_refreshed.append("vector_manager")
                    logger.info("Refreshed vector manager configuration")

        except (AttributeError, KeyError) as service_error:
            logger.warning("Failed to refresh some services after config update: %s", service_error)

        logger.info("Successfully updated config '%s': %s -> %s", config_name, old_value, config_obj.value)

        return {
            "message": f"Config '{config_name}' updated successfully",
            "old_value": old_value,
            "new_value": config_obj.value,
            "updated_in_memory": True,
            "services_refreshed": services_refreshed
        }
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Config '{config_name}' not found") from exc
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid value type: {e}") from e

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

@router.post("/config/test-update")
async def test_config_update_flow(request: Request):
    """설정 업데이트 흐름 테스트용 API"""
    try:
        config_composer = get_config_composer(request)

        # 현재 app.state 상태 확인
        state_info = {
            "has_config": hasattr(request.app.state, 'config'),
            "has_config_composer": hasattr(request.app.state, 'config_composer'),
            "has_rag_service": hasattr(request.app.state, 'rag_service'),
            "has_vector_manager": hasattr(request.app.state, 'vector_manager'),
            "config_categories": list(request.app.state.config.keys()) if hasattr(request.app.state, 'config') else [],
            "all_configs_count": len(config_composer.all_configs)
        }

        # 설정 객체들의 메모리 주소 확인 (참조가 같은지 확인)
        memory_refs = {}
        if hasattr(request.app.state, 'config') and request.app.state.config:
            for category_name, category_config in request.app.state.config.items():
                if category_name != "all_configs":
                    memory_refs[category_name] = {
                        "app_state_id": id(category_config),
                        "composer_id": id(config_composer.config_categories.get(category_name, None))
                    }

        return {
            "message": "Config update flow test completed",
            "app_state_info": state_info,
            "memory_references": memory_refs,
            "reference_consistency": all(
                ref["app_state_id"] == ref["composer_id"]
                for ref in memory_refs.values()
                if ref["composer_id"] is not None
            )
        }
    except Exception as e:
        logger.error("Config test failed: %s", e)
        return {"error": f"Config test failed: {e}"}

@router.post("/config/models/list")
async def get_models_list(request: Request):
    """모든 모델 관련 설정 정보 반환"""
    config_composer = get_config_composer(request)

    if not config_composer:
        raise HTTPException(status_code=500, detail="Configuration composer not available")

    openai_config = config_composer.get_config_by_category_name("openai").get_config_summary()
    vllm_config = config_composer.get_config_by_category_name("vllm").get_config_summary()

    result = []

    if not openai_config:
        openai_api_key = None
        openai_url = None
        print("OpenAI config not found")
    else:
        openai_api_key = openai_config.get("configs", {}).get("OPENAI_API_KEY", {}).get("current_value", "")
        openai_url = openai_config.get("configs", {}).get("OPENAI_API_BASE_URL", {}).get("current_value", "")

        openai_models = [
            "gpt-4o-2024-11-20",
            "gpt-4o-mini-2024-07-18",
            "gpt-4.1-2025-04-14",
            "gpt-4.1-mini-2025-04-14"
        ]

        temperature_default = openai_config.get("configs", {}).get("OPENAI_TEMPERATURE_DEFAULT", {}).get("current_value", 0.7)
        max_tokens_default = openai_config.get("configs", {}).get("OPENAI_MAX_TOKENS_DEFAULT", {}).get("current_value", 1000)

        for model in openai_models:
            result.append({
                "provider": "OpenAI",
                "api_key": openai_api_key,
                "api_base_url": openai_url,
                "model": model,
                "temperature_default": temperature_default,
                "max_tokens_default": max_tokens_default
            })

    if not vllm_config:
        vllm_model = None
        vllm_url = None
        print("VLLM config not found")
    else:
        vllm_model = vllm_config.get("configs", {}).get("VLLM_MODEL_NAME", {}).get("current_value", "")
        vllm_url = vllm_config.get("configs", {}).get("VLLM_API_BASE_URL", {}).get("current_value", "")
        result.append({
            "provider": "vLLM",
            "api_key": "",
            "api_base_url": vllm_url,
            "model": vllm_model,
            "temperature_default": vllm_config.get("configs", {}).get("VLLM_TEMPERATURE_DEFAULT", {}).get("current_value", 0.7),
            "max_tokens_default": vllm_config.get("configs", {}).get("VLLM_MAX_TOKENS_DEFAULT", {}).get("current_value", 512)
        })

    return {"result": result}

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

# Health & Status Endpoints
@router.get("/docs/health")
async def health_check(request: Request):
    """RAG 시스템 연결 상태 확인"""
    try:
        rag_service = get_rag_service(request)

        health_status = {
            "qdrant_client": rag_service.vector_manager.is_connected(),
            "embeddings_client": bool(rag_service.embeddings_client),
            "embedding_provider": rag_service.config.EMBEDDING_PROVIDER.value
        }

        if rag_service.vector_manager.is_connected():
            collections = rag_service.vector_manager.list_collections()
            health_status.update({
                "collections_count": len(collections["collections"]),
                "qdrant_status": "connected"
            })
        else:
            health_status["qdrant_status"] = "disconnected"

        # 임베딩 클라이언트 상태 상세 확인
        if rag_service.embeddings_client:
            try:
                is_available = await rag_service.embeddings_client.is_available()
                health_status["embeddings_status"] = "available" if is_available else "unavailable"
                health_status["embeddings_available"] = is_available
            except Exception as e:
                health_status["embeddings_status"] = "error"
                health_status["embeddings_error"] = str(e)
                health_status["embeddings_available"] = False
        else:
            health_status["embeddings_status"] = "not_initialized"
            health_status["embeddings_available"] = False

        overall_status = "healthy" if all([
            rag_service.vector_manager.is_connected(),
            rag_service.embeddings_client,
            health_status.get("embeddings_available", False)
        ]) else "partial"

        return {
            "status": overall_status,
            "message": "RAG system status check",
            "components": health_status
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "message": f"Health check failed: {e}"
        }

# Configuration Endpoints
@router.get("/docs/config")
async def get_rag_config(request: Request):
    """현재 RAG 시스템 설정 조회"""
    config_composer = get_config_composer(request)

    try:
        if config_composer:
            vectordb_config = config_composer.get_config_by_category_name("vectordb")

            return {
                "vectordb": {
                    "host": vectordb_config.QDRANT_HOST.value,
                    "port": vectordb_config.QDRANT_PORT.value,
                    "use_grpc": vectordb_config.QDRANT_USE_GRPC.value,
                    "grpc_port": vectordb_config.QDRANT_GRPC_PORT.value,
                    "collection_name": vectordb_config.COLLECTION_NAME.value,
                    "vector_dimension": vectordb_config.VECTOR_DIMENSION.value,
                    "replicas": vectordb_config.REPLICAS.value,
                    "shards": vectordb_config.SHARDS.value
                },
                "embedding": {
                    "provider": vectordb_config.EMBEDDING_PROVIDER.value,
                    "auto_detect_dimension": vectordb_config.AUTO_DETECT_EMBEDDING_DIM.value,
                    "openai": {
                        "api_key_configured": bool(vectordb_config.get_openai_api_key()),
                        "model": vectordb_config.OPENAI_EMBEDDING_MODEL.value
                    },
                    "huggingface": {
                        "model_name": vectordb_config.HUGGINGFACE_MODEL_NAME.value,
                        "api_key_configured": bool(vectordb_config.HUGGINGFACE_API_KEY.value)
                    },
                    "custom_http": {
                        "url": vectordb_config.CUSTOM_EMBEDDING_URL.value,
                        "model": vectordb_config.CUSTOM_EMBEDDING_MODEL.value,
                        "api_key_configured": bool(vectordb_config.CUSTOM_EMBEDDING_API_KEY.value)
                    }
                }
            }
        else:
            raise HTTPException(status_code=500, detail="Configuration not available")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get config: {str(e)}")

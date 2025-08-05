"""
Embedding 컨트롤러

임베딩 관련 API 엔드포인트를 제공합니다.
임베딩 제공자 관리, 상태 확인, 테스트 등의 기능을 담당합니다.
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from service.embedding import EmbeddingFactory
import logging

logger = logging.getLogger("embedding-controller")
router = APIRouter(prefix="/api/embedding", tags=["embedding"])

# Pydantic Models
class EmbeddingProviderSwitchRequest(BaseModel):
    new_provider: str

class EmbeddingTestRequest(BaseModel):
    query_text: str = "Hello, world!"

def get_embedding_client(request: Request):
    """임베딩 클라이언트 의존성 주입"""
    if hasattr(request.app.state, 'rag_service') and request.app.state.rag_service:
        return request.app.state.rag_service.embeddings_client
    else:
        raise HTTPException(status_code=500, detail="Embedding client not available")

def get_rag_service(request: Request):
    """RAG 서비스 의존성 주입"""
    if hasattr(request.app.state, 'rag_service') and request.app.state.rag_service:
        return request.app.state.rag_service
    else:
        raise HTTPException(status_code=500, detail="RAG service not available")

def get_config(request: Request):
    """설정 의존성 주입"""
    config_composer = request.app.state.config_composer
    if config_composer:
        return config_composer.get_all_config()
    else:
        raise HTTPException(status_code=500, detail="Configuration not available")

# Embedding Provider Management Endpoints
@router.get("/providers")
async def get_embedding_providers():
    """사용 가능한 임베딩 제공자 목록 조회"""
    try:
        providers = EmbeddingFactory.get_available_providers()
        return {
            "providers": providers,
            "total": len(providers)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get providers: {str(e)}")

@router.get("/test")
async def test_embedding_providers(request: Request):
    """모든 임베딩 제공자 테스트"""
    try:
        config = get_config(request)
        results = await EmbeddingFactory.test_all_providers(config["vectordb"])
        return {
            "test_results": results,
            "current_provider": config["vectordb"].EMBEDDING_PROVIDER.value
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to test providers: {str(e)}")

@router.get("/status")
async def get_embedding_status(request: Request):
    """현재 임베딩 클라이언트 상태 조회"""
    try:
        rag_service = get_rag_service(request)
        embedding_client = get_embedding_client(request)

        status_info = rag_service.get_embedding_status()

        # 실제 사용 가능성 체크 (비동기)
        if embedding_client:
            try:
                is_available = await embedding_client.is_available()
                status_info["available"] = is_available
            except Exception as e:
                status_info["available"] = False
                status_info["availability_error"] = str(e)

        return status_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get embedding status: {str(e)}")

@router.post("/test-query")
async def test_embedding_query(request: Request, test_request: EmbeddingTestRequest):
    """임베딩 생성 테스트"""
    try:
        rag_service = get_rag_service(request)
        config = get_config(request)

        # 자동 복구 로직이 포함된 임베딩 생성 (debug=True로 상세 로그 출력)
        print(rag_service)
        embedding = await rag_service.generate_query_embedding(test_request.query_text, debug=True)

        return {
            "query_text": test_request.query_text,
            "embedding_dimension": len(embedding),
            "embedding_preview": embedding[:5],  # 처음 5개 값만 미리보기
            "provider": config["vectordb"].EMBEDDING_PROVIDER.value,
            "success": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to test embedding: {str(e)}")

@router.post("/reload")
async def reload_embedding_client(request: Request):
    """임베딩 클라이언트 강제 재로드"""
    try:
        rag_service = get_rag_service(request)

        success = await rag_service.reload_embeddings_client()

        if success:
            provider_info = rag_service.embeddings_client.get_provider_info()
            return {
                "success": True,
                "message": "Embedding client reloaded successfully",
                "provider": rag_service.config.EMBEDDING_PROVIDER.value,
                "provider_info": provider_info
            }
        else:
            return {
                "success": False,
                "message": "Failed to reload embedding client",
                "provider": rag_service.config.EMBEDDING_PROVIDER.value
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload embedding client: {str(e)}")

@router.post("/switch-provider")
async def switch_embedding_provider(request: Request, switch_request: EmbeddingProviderSwitchRequest):
    """임베딩 제공자 변경"""
    try:
        rag_service = get_rag_service(request)
        config = get_config(request)
        new_provider = switch_request.new_provider

        # OpenAI 설정과 VectorDB 설정 연결 확인 및 재설정
        if "vectordb" in config and "openai" in config:
            config["vectordb"].set_openai_config(config["openai"])

        # 제공자 변경 시도
        old_provider = config["vectordb"].EMBEDDING_PROVIDER.value
        success = config["vectordb"].switch_embedding_provider(new_provider)

        if not success:
            # 실패 원인을 더 자세히 분석
            error_msg = f"Cannot switch to provider '{new_provider}'. "
            if new_provider.lower() == "openai":
                openai_key = config["vectordb"].get_openai_api_key()
                if not openai_key:
                    error_msg += "OpenAI API key is not configured."
                else:
                    error_msg += "OpenAI configuration validation failed."
            elif new_provider.lower() == "custom_http":
                custom_url = config["vectordb"].CUSTOM_EMBEDDING_URL.value
                if not custom_url or custom_url == "http://localhost:8000/v1":
                    error_msg += "Custom HTTP URL is not properly configured."

            raise HTTPException(status_code=400, detail=error_msg)

        # RAG 서비스의 임베딩 클라이언트 재로드
        reload_success = await rag_service.reload_embeddings_client()

        if reload_success:
            provider_info = rag_service.embeddings_client.get_provider_info()
            return {
                "success": True,
                "message": f"Embedding provider switched from {old_provider} to {new_provider}",
                "old_provider": old_provider,
                "new_provider": new_provider,
                "provider_info": provider_info
            }
        else:
            # 실패한 경우 원래 제공자로 롤백
            config["vectordb"].switch_embedding_provider(old_provider)
            await rag_service.reload_embeddings_client()

            raise HTTPException(
                status_code=500,
                detail=f"Failed to initialize new provider '{new_provider}'. Rolled back to '{old_provider}'"
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to switch embedding provider: {str(e)}")

@router.post("/auto-switch")
async def auto_switch_embedding_provider(request: Request):
    """자동으로 최적의 임베딩 제공자로 전환"""
    try:
        rag_service = get_rag_service(request)
        config = get_config(request)

        # OpenAI 설정과 VectorDB 설정 연결 확인 및 재설정
        if "vectordb" in config and "openai" in config:
            config["vectordb"].set_openai_config(config["openai"])

        old_provider = config["vectordb"].EMBEDDING_PROVIDER.value

        # 최적의 제공자로 자동 전환 시도
        switched = config["vectordb"].check_and_switch_to_best_provider()

        if switched:
            new_provider = config["vectordb"].EMBEDDING_PROVIDER.value

            # 클라이언트 재로드
            reload_success = await rag_service.reload_embeddings_client()

            if reload_success:
                provider_info = rag_service.embeddings_client.get_provider_info()
                return {
                    "success": True,
                    "switched": True,
                    "message": f"Auto-switched embedding provider from {old_provider} to {new_provider}",
                    "old_provider": old_provider,
                    "new_provider": new_provider,
                    "provider_info": provider_info
                }
            else:
                # 실패한 경우 원래 제공자로 롤백
                config["vectordb"].switch_embedding_provider(old_provider)
                await rag_service.reload_embeddings_client()

                return {
                    "success": False,
                    "switched": False,
                    "message": f"Failed to initialize new provider '{new_provider}'. Rolled back to '{old_provider}'",
                    "old_provider": old_provider,
                    "new_provider": old_provider
                }
        else:
            return {
                "success": True,
                "switched": False,
                "message": f"Current provider '{old_provider}' is already optimal",
                "current_provider": old_provider
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to auto-switch embedding provider: {str(e)}")

@router.get("/config-status")
async def get_embedding_config_status(request: Request):
    """임베딩 설정 상태 조회"""
    try:
        rag_service = get_rag_service(request)
        config_status = rag_service.config.get_embedding_provider_status()

        # 현재 클라이언트 상태 추가
        if rag_service.embeddings_client:
            try:
                is_available = await rag_service.embeddings_client.is_available()
                provider_info = rag_service.embeddings_client.get_provider_info()
                config_status.update({
                    "client_initialized": True,
                    "client_available": is_available,
                    "provider_info": provider_info
                })
            except Exception as e:
                config_status.update({
                    "client_initialized": True,
                    "client_available": False,
                    "client_error": str(e)
                })
        else:
            config_status.update({
                "client_initialized": False,
                "client_available": False
            })

        return config_status
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get config status: {str(e)}")

@router.get("/debug/info")
async def get_embedding_debug_info(request: Request):
    """디버깅을 위한 임베딩 상세 정보 조회"""
    try:
        rag_service = get_rag_service(request)
        vectordb_config = rag_service.config

        debug_info = {
            "current_provider": vectordb_config.EMBEDDING_PROVIDER.value,
            "auto_detect_dimension": vectordb_config.AUTO_DETECT_EMBEDDING_DIM.value,
            "vector_dimension": vectordb_config.VECTOR_DIMENSION.value,
            "client_initialized": bool(rag_service.embeddings_client),
            "dependencies": {},
            "provider_configs": {
                "openai": {
                    "api_key_configured": bool(vectordb_config.get_openai_api_key()),
                    "api_key_length": len(vectordb_config.get_openai_api_key()) if vectordb_config.get_openai_api_key() else 0,
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

        # 의존성 확인
        try:
            import sentence_transformers
            debug_info["dependencies"]["sentence_transformers"] = {
                "available": True,
                "version": getattr(sentence_transformers, "__version__", "unknown")
            }
        except ImportError as e:
            debug_info["dependencies"]["sentence_transformers"] = {
                "available": False,
                "error": str(e)
            }

        try:
            import openai
            debug_info["dependencies"]["openai"] = {
                "available": True,
                "version": getattr(openai, "__version__", "unknown")
            }
        except ImportError as e:
            debug_info["dependencies"]["openai"] = {
                "available": False,
                "error": str(e)
            }

        try:
            import aiohttp
            debug_info["dependencies"]["aiohttp"] = {
                "available": True,
                "version": getattr(aiohttp, "__version__", "unknown")
            }
        except ImportError as e:
            debug_info["dependencies"]["aiohttp"] = {
                "available": False,
                "error": str(e)
            }

        # 클라이언트 상세 정보
        if rag_service.embeddings_client:
            try:
                client_info = rag_service.embeddings_client.get_provider_info()
                debug_info["client_info"] = client_info

                # 기본 사용 가능성 체크
                # basic_available = rag_service._check_client_basic_availability(rag_service.embeddings_client)
                # debug_info["basic_availability"] = basic_available

                # 실제 사용 가능성 체크 (비동기)
                try:
                    full_available = await rag_service.embeddings_client.is_available()
                    debug_info["full_availability"] = full_available
                except Exception as e:
                    debug_info["full_availability"] = False
                    debug_info["availability_error"] = str(e)

            except Exception as e:
                debug_info["client_error"] = str(e)

        return debug_info

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get debug info: {str(e)}")

@router.get("/health")
async def embedding_health_check(request: Request):
    """임베딩 시스템 상태 확인"""
    try:
        rag_service = get_rag_service(request)

        health_status = {
            "embeddings_client": bool(rag_service.embeddings_client),
            "embedding_provider": rag_service.config.EMBEDDING_PROVIDER.value
        }

        # 임베딩 클라이언트 상태 상세 확인
        if rag_service.embeddings_client:
            try:
                is_available = await rag_service.embeddings_client.is_available()
                health_status["embeddings_status"] = "available" if is_available else "unavailable"
                health_status["embeddings_available"] = is_available

                # 제공자 정보 추가
                provider_info = rag_service.embeddings_client.get_provider_info()
                health_status["provider_info"] = provider_info

            except Exception as e:
                health_status["embeddings_status"] = "error"
                health_status["embeddings_error"] = str(e)
                health_status["embeddings_available"] = False
        else:
            health_status["embeddings_status"] = "not_initialized"
            health_status["embeddings_available"] = False

        overall_status = "healthy" if health_status.get("embeddings_available", False) else "unhealthy"

        return {
            "status": overall_status,
            "message": "Embedding system status check",
            "components": health_status
        }
    except Exception as e:
        logger.error("Embedding health check failed: %s", e)
        return {
            "status": "unhealthy",
            "message": f"Embedding health check failed: {e}"
        }

"""
Embedding 컨트롤러

임베딩 관련 API 엔드포인트를 제공합니다.
임베딩 제공자 관리, 상태 확인, 테스트 등의 기능을 담당합니다.
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
import logging
import gc
from controller.helper.singletonHelper import get_embedding_client, get_rag_service, get_config_composer
from service.embedding.embedding_factory import EmbeddingFactory
from controller.rag.helper import safely_replace_embedding_client

logger = logging.getLogger("embedding-controller")
router = APIRouter(prefix="/embedding", tags=["embedding"])

class EmbeddingProviderSwitchRequest(BaseModel):
    new_provider: str

class EmbeddingTestRequest(BaseModel):
    query_text: str = "Hello, world!"

@router.post("/test-query")
async def test_embedding_query(request: Request, test_request: EmbeddingTestRequest):
    """임베딩 생성 테스트"""
    try:
        rag_service = get_rag_service(request)
        config_composer = get_config_composer(request)
        embedding = await rag_service.generate_query_embedding(test_request.query_text, debug=True)

        return {
            "query_text": test_request.query_text,
            "embedding_dimension": len(embedding),
            "embedding_preview": embedding[:5],
            "provider": config_composer.get_config_by_name("EMBEDDING_PROVIDER").value,
            "success": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to test embedding: {str(e)}")

@router.post("/switch-provider")
async def switch_embedding_provider(request: Request, switch_request: EmbeddingProviderSwitchRequest):
    """임베딩 제공자 변경"""
    try:
        config_composer = get_config_composer(request)
        new_provider = switch_request.new_provider

        embedding_config = config_composer.get_config_by_category_name('embedding')

        old_provider = embedding_config.EMBEDDING_PROVIDER.value
        update_result = config_composer.update_config('EMBEDDING_PROVIDER', new_provider)

        if not update_result:
            error_msg = f"Cannot switch to provider '{new_provider}'. "
            if new_provider.lower() == "openai":
                openai_key = config_composer.get_config_by_name("OPENAI_API_KEY").value
                if not openai_key or openai_key == "" or len(openai_key) < 10:
                    error_msg += "OpenAI API key is not configured."
                else:
                    error_msg += "OpenAI configuration validation failed."
            elif new_provider.lower() == "custom_http":
                custom_url = config_composer.get_config_by_name("CUSTOM_EMBEDDING_URL").value
                if not custom_url or custom_url == "http://localhost:8000/v1" or len(custom_url) < 10:
                    error_msg += "Custom HTTP URL is not properly configured."

            raise HTTPException(status_code=400, detail=error_msg)

        embedding_client = EmbeddingFactory.create_embedding_client(config_composer)
        safely_replace_embedding_client(request, embedding_client)

        if embedding_client:
            provider_info = embedding_client.get_provider_info()
            return {
                "success": True,
                "message": f"Embedding provider switched from {old_provider} to {new_provider}",
                "old_provider": old_provider,
                "new_provider": new_provider,
                "provider_info": provider_info
            }
        else:
            config_composer.update_config('EMBEDDING_PROVIDER', old_provider)
            embedding_client = EmbeddingFactory.create_embedding_client(config_composer)
            safely_replace_embedding_client(request, embedding_client)

            raise HTTPException(
                status_code=500,
                detail=f"Failed to initialize new provider '{new_provider}'. Rolled back to '{old_provider}'"
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to switch embedding provider: {str(e)}")

@router.get("/config-status")
async def get_embedding_config_status(request: Request):
    """임베딩 설정 상태 조회"""
    try:
        embedding_client = get_embedding_client(request)
        config_status = {
            "client_initialized": False,
            "client_available": False,
            "provider_info": {},
        }

        if embedding_client:
            try:
                is_available = await embedding_client.is_available()
                provider_info = embedding_client.get_provider_info()
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

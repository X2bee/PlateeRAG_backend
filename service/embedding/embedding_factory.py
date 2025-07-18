"""
임베딩 팩토리
설정에 따라 적절한 임베딩 클라이언트를 생성
"""

from typing import Dict, Any
import logging
from service.embedding.base_embedding import BaseEmbedding
from service.embedding.openai_embedding import OpenAIEmbedding
from service.embedding.huggingface_embedding import HuggingFaceEmbedding
from service.embedding.custom_http_embedding import CustomHTTPEmbedding

logger = logging.getLogger("embeddings.factory")

class EmbeddingFactory:
    """임베딩 클라이언트 팩토리"""
    
    PROVIDERS = {
        "openai": OpenAIEmbedding,
        "huggingface": HuggingFaceEmbedding,
        "custom_http": CustomHTTPEmbedding
    }
    
    @classmethod
    def create_embedding_client(cls, vectordb_config) -> BaseEmbedding:
        """
        설정에 따라 임베딩 클라이언트 생성
        
        Args:
            vectordb_config: VectorDBConfig 객체
            
        Returns:
            BaseEmbedding 구현체
        """
        provider = vectordb_config.EMBEDDING_PROVIDER.value.lower()
        
        if provider not in cls.PROVIDERS:
            available_providers = list(cls.PROVIDERS.keys())
            raise ValueError(f"Unsupported embedding provider: {provider}. Available: {available_providers}")
        
        # 제공자별 설정 준비
        config = cls._prepare_config(provider, vectordb_config)
        
        # 클라이언트 생성
        try:
            embedding_class = cls.PROVIDERS[provider]
            client = embedding_class(config)
            
            logger.info(f"Created {provider} embedding client")
            return client
            
        except ImportError as e:
            logger.error(f"Missing dependencies for {provider} embedding client: {e}")
            raise ValueError(f"Cannot create {provider} embedding client. Missing dependencies: {e}")
        except Exception as e:
            logger.error(f"Failed to create {provider} embedding client: {e}")
            raise
    
    @classmethod
    def _prepare_config(cls, provider: str, vectordb_config) -> Dict[str, Any]:
        """제공자별 설정 준비"""
        
        if provider == "openai":
            return {
                "api_key": vectordb_config.get_openai_api_key(),
                "model": vectordb_config.OPENAI_EMBEDDING_MODEL.value
            }
        
        elif provider == "huggingface":
            return {
                "model_name": vectordb_config.HUGGINGFACE_MODEL_NAME.value,
                "api_key": vectordb_config.HUGGINGFACE_API_KEY.value
            }
        
        elif provider == "custom_http":
            return {
                "url": vectordb_config.CUSTOM_EMBEDDING_URL.value,
                "api_key": vectordb_config.CUSTOM_EMBEDDING_API_KEY.value,
                "model": vectordb_config.CUSTOM_EMBEDDING_MODEL.value
            }
        
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    @classmethod
    async def test_all_providers(cls, vectordb_config) -> Dict[str, Dict[str, Any]]:
        """
        모든 임베딩 제공자 테스트
        
        Args:
            vectordb_config: VectorDBConfig 객체
            
        Returns:
            제공자별 테스트 결과
        """
        results = {}
        
        for provider_name in cls.PROVIDERS:
            try:
                # 임시로 제공자 변경
                original_provider = vectordb_config.EMBEDDING_PROVIDER.value
                vectordb_config.EMBEDDING_PROVIDER.value = provider_name
                
                # 클라이언트 생성 및 테스트
                client = cls.create_embedding_client(vectordb_config)
                is_available = await client.is_available()
                provider_info = client.get_provider_info()
                provider_info["available"] = is_available
                
                results[provider_name] = provider_info
                
                # 원래 제공자로 복원
                vectordb_config.EMBEDDING_PROVIDER.value = original_provider
                
            except Exception as e:
                results[provider_name] = {
                    "provider": provider_name,
                    "available": False,
                    "error": str(e)
                }
                
                # 원래 제공자로 복원
                vectordb_config.EMBEDDING_PROVIDER.value = original_provider
        
        return results
    
    @classmethod
    def get_available_providers(cls) -> Dict[str, str]:
        """
        사용 가능한 제공자 목록 반환
        
        Returns:
            제공자명과 설명
        """
        return {
            "openai": "OpenAI 임베딩 API (text-embedding-3-small, text-embedding-ada-002 등)",
            "huggingface": "HuggingFace sentence-transformers 모델 (로컬 실행)",
            "custom_http": "Custom HTTP API (vLLM, FastAPI 등 OpenAI 호환 서버)"
        } 
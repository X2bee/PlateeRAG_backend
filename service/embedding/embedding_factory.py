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

    _instance = None
    _last_config_hash = None

    @classmethod
    def create_embedding_client(cls, config_composer) -> BaseEmbedding:
        embedding_config = config_composer.get_config_by_category_name("embedding")
        provider = embedding_config.EMBEDDING_PROVIDER.value.lower()

        # 설정 해시 생성 (설정이 변경되었는지 확인용)
        config_hash = cls._generate_config_hash(provider, config_composer)

        # 기존 인스턴스가 있고 설정이 동일하면 재사용
        if cls._instance is not None and cls._last_config_hash == config_hash:
            logger.info("Reusing existing embedding client instance")
            return cls._instance

        # 기존 인스턴스가 있지만 설정이 변경된 경우
        if cls._instance is not None:
            logger.info("Configuration changed, replacing embedding client")
            cls._instance = None

        if provider not in cls.PROVIDERS:
            available_providers = list(cls.PROVIDERS.keys())
            raise ValueError(f"Unsupported embedding provider: {provider}. Available: {available_providers}")

        config = cls._prepare_config(provider, config_composer)

        try:
            embedding_class = cls.PROVIDERS[provider]
            client = embedding_class(config)

            logger.info(f"Created {provider} embedding client")

            if embedding_config.AUTO_DETECT_EMBEDDING_DIM.value:
                dimension = client.get_embedding_dimension()
                config_composer.update_config("QDRANT_VECTOR_DIMENSION", dimension)

            # 새 인스턴스와 설정 해시 저장
            cls._instance = client
            cls._last_config_hash = config_hash
            return client

        except ImportError as e:
            logger.error(f"Missing dependencies for {provider} embedding client: {e}")
            raise ValueError(f"Cannot create {provider} embedding client. Missing dependencies: {e}")
        except Exception as e:
            logger.error(f"Failed to create {provider} embedding client: {e}")
            raise

    @classmethod
    def _generate_config_hash(cls, provider: str, config_composer) -> str:
        """설정 변경 감지를 위한 해시 생성"""
        embedding_config = config_composer.get_config_by_category_name("embedding")

        if provider == "openai":
            config_str = f"{provider}:{config_composer.get_config_by_name('OPENAI_API_KEY').value}:{embedding_config.OPENAI_EMBEDDING_MODEL_NAME.value}"
        elif provider == "huggingface":
            config_str = f"{provider}:{embedding_config.HUGGINGFACE_EMBEDDING_MODEL_NAME.value}:{config_composer.get_config_by_name('HUGGING_FACE_HUB_TOKEN').value}:{embedding_config.HUGGINGFACE_EMBEDDING_MODEL_DEVICE.value}"
        elif provider == "custom_http":
            config_str = f"{provider}:{embedding_config.CUSTOM_EMBEDDING_URL.value}:{embedding_config.CUSTOM_EMBEDDING_API_KEY.value}:{embedding_config.CUSTOM_EMBEDDING_MODEL_NAME.value}"
        else:
            config_str = provider

        return str(hash(config_str))

    @classmethod
    def _prepare_config(cls, provider: str, config_composer) -> Dict[str, Any]:
        embedding_config = config_composer.get_config_by_category_name("embedding")

        if provider == "openai":
            return {
                "api_key": config_composer.get_config_by_name("OPENAI_API_KEY").value,
                "model": embedding_config.OPENAI_EMBEDDING_MODEL_NAME.value
            }

        elif provider == "huggingface":
            return {
                "model_name": embedding_config.HUGGINGFACE_EMBEDDING_MODEL_NAME.value,
                "api_key": config_composer.get_config_by_name("HUGGING_FACE_HUB_TOKEN").value,
                "model_device": embedding_config.HUGGINGFACE_EMBEDDING_MODEL_DEVICE.value,
            }

        elif provider == "custom_http":
            return {
                "url": embedding_config.CUSTOM_EMBEDDING_URL.value,
                "api_key": embedding_config.CUSTOM_EMBEDDING_API_KEY.value,
                "model": embedding_config.CUSTOM_EMBEDDING_MODEL_NAME.value
            }

        else:
            raise ValueError(f"Unknown provider: {provider}")

    @classmethod
    def get_available_providers(cls) -> Dict[str, str]:
        """
        사용 가능한 제공자 목록 반환

        Returns:
            제공자명과 설명
        """
        return {
            "openai": "",
            "huggingface": "",
            "custom_http": ""
        }

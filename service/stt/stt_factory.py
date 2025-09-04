"""
STT 팩토리
설정에 따라 적절한 STT 클라이언트를 생성
"""

from typing import Dict, Any
import logging
from service.stt.base_stt import BaseSTT
from service.stt.huggingface_stt import HuggingFaceSTT

logger = logging.getLogger("stt.factory")

class STTFactory:
    """STT 클라이언트 팩토리"""

    PROVIDERS = {
        "huggingface": HuggingFaceSTT,
        # 추후 OpenAI STT 등 추가 가능
        # "openai": OpenAISTT,
    }

    _instance = None
    _last_config_hash = None

    @classmethod
    def create_stt_client(cls, config_composer) -> BaseSTT:
        stt_config = config_composer.get_config_by_category_name("stt")
        provider = stt_config.STT_PROVIDER.value.lower()

        # 설정 해시 생성 (설정이 변경되었는지 확인용)
        config_hash = cls._generate_config_hash(provider, config_composer)

        # 기존 인스턴스가 있고 설정이 동일하면 재사용
        if cls._instance is not None and cls._last_config_hash == config_hash:
            logger.info("Reusing existing STT client instance")
            return cls._instance

        # 기존 인스턴스가 있지만 설정이 변경된 경우
        if cls._instance is not None:
            logger.info("Configuration changed, replacing STT client")
            cls._instance = None

        if provider not in cls.PROVIDERS:
            available_providers = list(cls.PROVIDERS.keys())
            raise ValueError(f"Unsupported STT provider: {provider}. Available: {available_providers}")

        config = cls._prepare_config(provider, config_composer)

        try:
            stt_class = cls.PROVIDERS[provider]
            client = stt_class(config)

            logger.info(f"Created {provider} STT client")

            # 새 인스턴스와 설정 해시 저장
            cls._instance = client
            cls._last_config_hash = config_hash
            return client

        except ImportError as e:
            logger.error(f"Missing dependencies for {provider} STT client: {e}")
            raise ValueError(f"Cannot create {provider} STT client. Missing dependencies: {e}")
        except Exception as e:
            logger.error(f"Failed to create {provider} STT client: {e}")
            raise

    @classmethod
    def _generate_config_hash(cls, provider: str, config_composer) -> str:
        """설정 변경 감지를 위한 해시 생성"""
        stt_config = config_composer.get_config_by_category_name("stt")

        if provider == "huggingface":
            config_str = f"{provider}:{stt_config.HUGGINGFACE_STT_MODEL_NAME.value}:{config_composer.get_config_by_name('HUGGING_FACE_HUB_TOKEN').value}:{stt_config.HUGGINGFACE_STT_MODEL_DEVICE.value}"
        elif provider == "openai":
            config_str = f"{provider}:{config_composer.get_config_by_name('OPENAI_API_KEY').value}:{stt_config.OPENAI_STT_MODEL_NAME.value}"
        else:
            config_str = provider

        return str(hash(config_str))

    @classmethod
    def _prepare_config(cls, provider: str, config_composer) -> Dict[str, Any]:
        stt_config = config_composer.get_config_by_category_name("stt")

        if provider == "huggingface":
            return {
                "model_name": stt_config.HUGGINGFACE_STT_MODEL_NAME.value,
                "api_key": config_composer.get_config_by_name("HUGGING_FACE_HUB_TOKEN").value,
                "model_device": stt_config.HUGGINGFACE_STT_MODEL_DEVICE.value,
            }

        elif provider == "openai":
            return {
                "api_key": config_composer.get_config_by_name("OPENAI_API_KEY").value,
                "model": stt_config.OPENAI_STT_MODEL_NAME.value
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
            "huggingface": "HuggingFace Transformers STT",
            # "openai": "OpenAI Whisper STT"
        }

    @classmethod
    async def cleanup_instance(cls):
        """인스턴스 정리"""
        if cls._instance:
            try:
                await cls._instance.cleanup()
            except Exception as e:
                logger.warning(f"Error during STT client cleanup: {e}")
            finally:
                cls._instance = None
                cls._last_config_hash = None

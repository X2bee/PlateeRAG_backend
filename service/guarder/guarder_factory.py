"""
Guarder 팩토리
"""

from typing import Dict, Any
import logging
from service.guarder.base_guarder import BaseGuarder
from service.guarder.qwen3guard_guarder import Qwen3GuardGuarder

logger = logging.getLogger("guarder.factory")

class GuarderFactory:
    """Guarder 클라이언트 팩토리"""

    PROVIDERS = {
        "qwen3guard": Qwen3GuardGuarder,
    }

    _instance = None
    _last_config_hash = None

    @classmethod
    def create_guarder_client(cls, config_composer) -> BaseGuarder:
        guarder_config = config_composer.get_config_by_category_name("guarder")
        provider = guarder_config.GUARDER_PROVIDER.value.lower()

        # 설정 해시 생성 (설정이 변경되었는지 확인용)
        config_hash = cls._generate_config_hash(provider, config_composer)

        # 기존 인스턴스가 있고 설정이 동일하면 재사용
        if cls._instance is not None and cls._last_config_hash == config_hash:
            logger.info("Reusing existing Guarder client instance")
            return cls._instance

        # 기존 인스턴스가 있지만 설정이 변경된 경우
        if cls._instance is not None:
            logger.info("Configuration changed, replacing Guarder client")
            cls._instance = None

        if provider not in cls.PROVIDERS:
            available_providers = list(cls.PROVIDERS.keys())
            raise ValueError(f"Unsupported Guarder provider: {provider}. Available: {available_providers}")

        config = cls._prepare_config(provider, config_composer)

        try:
            guarder_class = cls.PROVIDERS[provider]
            client = guarder_class(config)

            logger.info("Created %s Guarder client", provider)

            # 새 인스턴스와 설정 해시 저장
            cls._instance = client
            cls._last_config_hash = config_hash
            return client

        except ImportError as e:
            logger.error("Missing dependencies for %s Guarder client: %s", provider, e)
            raise ValueError(f"Cannot create {provider} Guarder client. Missing dependencies: {e}") from e
        except (RuntimeError, OSError, ValueError) as e:
            logger.error("Failed to create %s Guarder client: %s", provider, e)
            raise

    @classmethod
    def _generate_config_hash(cls, provider: str, config_composer) -> str:
        """설정 변경 감지를 위한 해시 생성"""
        guarder_config = config_composer.get_config_by_category_name("guarder")

        if provider == "qwen3guard":
            config_str = f"{provider}:{guarder_config.QWEN3GUARD_MODEL_NAME.value}:{guarder_config.QWEN3GUARD_MODEL_DEVICE.value}"
            # API 키가 있는 경우 해시에 포함
            if hasattr(guarder_config, 'HUGGING_FACE_HUB_TOKEN'):
                hf_token = config_composer.get_config_by_name('HUGGING_FACE_HUB_TOKEN').value
                config_str += f":{hf_token}"
        else:
            config_str = provider

        return str(hash(config_str))

    @classmethod
    def _prepare_config(cls, provider: str, config_composer) -> Dict[str, Any]:
        guarder_config = config_composer.get_config_by_category_name("guarder")

        if provider == "qwen3guard":
            config = {
                "model_name": guarder_config.QWEN3GUARD_MODEL_NAME.value,
                "model_device": guarder_config.QWEN3GUARD_MODEL_DEVICE.value,
            }

            # HuggingFace API 키가 있는 경우 추가
            try:
                hf_token = config_composer.get_config_by_name('HUGGING_FACE_HUB_TOKEN').value
                if hf_token:
                    config["api_key"] = hf_token
            except (AttributeError, KeyError):
                logger.info("HUGGING_FACE_HUB_TOKEN not configured, using public access")

            return config

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
            "qwen3guard": "Qwen3Guard Text Moderation",
            # "openai_moderation": "OpenAI Moderation API",
            # "perspective": "Google Perspective API"
        }

    @classmethod
    async def cleanup_instance(cls):
        """인스턴스 정리"""
        if cls._instance:
            try:
                await cls._instance.cleanup()
            except (RuntimeError, AttributeError) as e:
                logger.warning("Error during Guarder client cleanup: %s", e)
            finally:
                cls._instance = None
                cls._last_config_hash = None

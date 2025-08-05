"""
Collection 관리 관련 설정
"""
from typing import Dict
from config.base_config import BaseConfig, PersistentConfig

class CollectionConfig(BaseConfig):
    """Collection 관리 관련 설정 관리"""

    def initialize(self) -> Dict[str, PersistentConfig]:
        """Collection 관련 설정들을 초기화"""

        # 이미지-텍스트 모델 제공자 설정
        self.IMAGE_TEXT_MODEL_PROVIDER = self.create_persistent_config(
            env_name="IMAGE_TEXT_MODEL_PROVIDER",
            config_path="collection.image_text_model_provider",
            default_value="openai"
        )

        # 이미지-텍스트 모델 Base URL
        self.IMAGE_TEXT_BASE_URL = self.create_persistent_config(
            env_name="IMAGE_TEXT_BASE_URL",
            config_path="collection.image_text_base_url",
            default_value="https://api.openai.com/v1"
        )

        # 이미지-텍스트 모델 API 키
        self.IMAGE_TEXT_API_KEY = self.create_persistent_config(
            env_name="IMAGE_TEXT_API_KEY",
            config_path="collection.image_text_api_key",
            default_value=""
        )

        # 이미지-텍스트 모델 이름
        self.IMAGE_TEXT_MODEL_NAME = self.create_persistent_config(
            env_name="IMAGE_TEXT_MODEL_NAME",
            config_path="collection.image_text_model_name",
            default_value="gpt-4-vision-preview"
        )

        # 온도 설정
        self.IMAGE_TEXT_TEMPERATURE = self.create_persistent_config(
            env_name="IMAGE_TEXT_TEMPERATURE",
            config_path="collection.image_text_temperature",
            default_value=0.7,
            type_converter=float
        )

        # 이미지 품질 설정
        self.IMAGE_QUALITY = self.create_persistent_config(
            env_name="IMAGE_QUALITY",
            config_path="collection.image_quality",
            default_value="auto"
        )

        self.IMAGE_TEXT_BATCH_SIZE = self.create_persistent_config(
            env_name="IMAGE_TEXT_BATCH_SIZE",
            config_path="collection.image_test_batch_size",
            default_value=1
        )

        # 제공자 검증
        valid_providers = ["openai", "vLLM", "no_model"]
        if self.IMAGE_TEXT_MODEL_PROVIDER.value not in valid_providers:
            self.logger.warning(
                f"Invalid image-text model provider '{self.IMAGE_TEXT_MODEL_PROVIDER.value}'. "
                f"Valid options: {', '.join(valid_providers)}. "
                f"Defaulting to 'openai'."
            )
            # 올바른 방법으로 값 설정
            self.IMAGE_TEXT_MODEL_PROVIDER.value = "openai"
            self.IMAGE_TEXT_MODEL_PROVIDER.save()

        # 환경변수에 설정 적용
        import os
        os.environ["IMAGE_TEXT_MODEL_PROVIDER"] = self.IMAGE_TEXT_MODEL_PROVIDER.value
        self.logger.info(f"Image-text model provider set to: {self.IMAGE_TEXT_MODEL_PROVIDER.value}")

        return self.configs

    def get_supported_formats(self) -> list:
        """지원되는 이미지 형식 반환"""
        return ["jpg", "jpeg", "png", "gif", "webp"]

    def get_max_image_size(self) -> int:
        """최대 이미지 크기 반환 (MB)"""
        return 10  # 기본값 10MB

    def get_model_info(self) -> Dict[str, any]:
        """현재 모델 정보 반환"""
        return {
            "provider": self.get_current_provider(),
            "model_name": self.IMAGE_TEXT_MODEL_NAME.value,
            "temperature": self.IMAGE_TEXT_TEMPERATURE.value,
            "image_quality": self.IMAGE_QUALITY.value,
            "image_text_batch_size" : self.IMAGE_TEXT_BATCH_SIZE.value,
            "supported_formats": self.get_supported_formats(),
            "max_image_size_mb": self.get_max_image_size()
        }

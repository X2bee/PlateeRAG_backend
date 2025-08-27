"""
DocumentProcessor 관리 관련 설정
"""
from typing import Dict
from config.base_config import BaseConfig, PersistentConfig

class DocumentProcessorConfig(BaseConfig):
    """DocumentProcessor 관리 관련 설정 관리"""

    def initialize(self) -> Dict[str, PersistentConfig]:
        """DocumentProcessor 관련 설정들을 초기화"""

        # 이미지-텍스트 모델 제공자 설정
        self.DOCUMENT_PROCESSOR_IMAGE_TEXT_MODEL_PROVIDER = self.create_persistent_config(
            env_name="DOCUMENT_PROCESSOR_IMAGE_TEXT_MODEL_PROVIDER",
            config_path="document_processor.image_text_model_provider",
            default_value="openai"
        )

        # 이미지-텍스트 모델 Base URL
        self.DOCUMENT_PROCESSOR_OPENAI_IMAGE_TEXT_BASE_URL = self.create_persistent_config(
            env_name="DOCUMENT_PROCESSOR_OPENAI_IMAGE_TEXT_BASE_URL",
            config_path="document_processor.openai.image_text_base_url",
            default_value="https://api.openai.com/v1"
        )

        # 이미지-텍스트 모델 API 키
        self.DOCUMENT_PROCESSOR_OPENAI_IMAGE_TEXT_API_KEY = self.create_persistent_config(
            env_name="DOCUMENT_PROCESSOR_OPENAI_IMAGE_TEXT_API_KEY",
            config_path="document_processor.openai.image_text_api_key",
            default_value=""
        )

        self.DOCUMENT_PROCESSOR_OPENAI_IMAGE_TEXT_MODEL_NAME = self.create_persistent_config(
            env_name="DOCUMENT_PROCESSOR_OPENAI_IMAGE_TEXT_MODEL_NAME",
            config_path="document_processor.openai.image_text_model_name",
            default_value="gpt-4.1-mini-2025-04-14"
        )

        # 이미지-텍스트 모델 Base URL
        self.DOCUMENT_PROCESSOR_VLLM_IMAGE_TEXT_BASE_URL = self.create_persistent_config(
            env_name="DOCUMENT_PROCESSOR_VLLM_IMAGE_TEXT_BASE_URL",
            config_path="document_processor.vllm.image_text_base_url",
            default_value=""
        )

        # 이미지-텍스트 모델 API 키
        self.DOCUMENT_PROCESSOR_VLLM_IMAGE_TEXT_API_KEY = self.create_persistent_config(
            env_name="DOCUMENT_PROCESSOR_VLLM_IMAGE_TEXT_API_KEY",
            config_path="document_processor.vllm.image_text_api_key",
            default_value=""
        )

        # 이미지-텍스트 모델 이름
        self.DOCUMENT_PROCESSOR_VLLM_IMAGE_TEXT_MODEL_NAME = self.create_persistent_config(
            env_name="DOCUMENT_PROCESSOR_VLLM_IMAGE_TEXT_MODEL_NAME",
            config_path="document_processor.vllm.image_text_model_name",
            default_value=""
        )

        # 온도 설정
        self.DOCUMENT_PROCESSOR_IMAGE_TEXT_TEMPERATURE = self.create_persistent_config(
            env_name="DOCUMENT_PROCESSOR_IMAGE_TEXT_TEMPERATURE",
            config_path="document_processor.image_text_temperature",
            default_value=0.7,
            type_converter=float
        )

        # 이미지 품질 설정
        self.DOCUMENT_PROCESSOR_IMAGE_QUALITY = self.create_persistent_config(
            env_name="DOCUMENT_PROCESSOR_IMAGE_QUALITY",
            config_path="document_processor.image_quality",
            default_value="auto"
        )

        self.DOCUMENT_PROCESSOR_IMAGE_TEXT_BATCH_SIZE = self.create_persistent_config(
            env_name="DOCUMENT_PROCESSOR_IMAGE_TEXT_BATCH_SIZE",
            config_path="document_processor.image_text_batch_size",
            default_value=1
        )

        return self.configs

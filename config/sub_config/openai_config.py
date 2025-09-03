"""
OpenAI API 관련 설정
"""
from typing import Dict
from config.base_config import BaseConfig, PersistentConfig

class OpenAIConfig(BaseConfig):
    """OpenAI API 관련 설정 관리"""

    def initialize(self) -> Dict[str, PersistentConfig]:
        """OpenAI 관련 설정들을 초기화"""

        # OpenAI API Key 설정
        self.API_KEY = self.create_persistent_config(
            env_name="OPENAI_API_KEY",
            config_path="openai.api_key",
            default_value="",
            file_path="openai_api_key.txt"
        )

        # 기본 모델 설정
        self.MODEL_DEFAULT = self.create_persistent_config(
            env_name="OPENAI_MODEL_DEFAULT",
            config_path="openai.model_default",
            default_value="gpt-4o-2024-11-20"
        )

        # API Base URL (프록시나 다른 엔드포인트 사용시)
        self.API_BASE_URL = self.create_persistent_config(
            env_name="OPENAI_API_BASE_URL",
            config_path="openai.api_base_url",
            default_value="https://api.openai.com/v1"
        )

        # 기본 온도 설정
        self.TEMPERATURE_DEFAULT = self.create_persistent_config(
            env_name="OPENAI_TEMPERATURE_DEFAULT",
            config_path="openai.temperature_default",
            default_value=0.7,
            type_converter=float
        )

        # 기본 최대 토큰 설정
        self.MAX_TOKENS_DEFAULT = self.create_persistent_config(
            env_name="OPENAI_MAX_TOKENS_DEFAULT",
            config_path="openai.max_tokens_default",
            default_value=1000,
            type_converter=int
        )

        # API 요청 타임아웃
        self.REQUEST_TIMEOUT = self.create_persistent_config(
            env_name="OPENAI_REQUEST_TIMEOUT",
            config_path="openai.request_timeout",
            default_value=30,
            type_converter=int
        )

        return self.configs

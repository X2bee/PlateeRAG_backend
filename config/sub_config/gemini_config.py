"""
Gemini API 관련 설정
"""
from typing import Dict
from config.base_config import BaseConfig, PersistentConfig

class GeminiConfig(BaseConfig):
    """Gemini API 관련 설정 관리"""

    def initialize(self) -> Dict[str, PersistentConfig]:
        """Gemini 관련 설정들을 초기화"""

        self.API_KEY = self.create_persistent_config(
            env_name="GEMINI_API_KEY",
            config_path="gemini.api_key",
            default_value="",
            file_path="gemini_api_key.txt"
        )

        self.MODEL_DEFAULT = self.create_persistent_config(
            env_name="GEMINI_MODEL_DEFAULT",
            config_path="gemini.model_default",
            default_value="gemini-2.5-flash"
        )

        self.API_BASE_URL = self.create_persistent_config(
            env_name="GEMINI_API_BASE_URL",
            config_path="gemini.api_base_url",
            default_value="https://generativelanguage.googleapis.com/v1beta"
        )

        self.TEMPERATURE_DEFAULT = self.create_persistent_config(
            env_name="GEMINI_TEMPERATURE_DEFAULT",
            config_path="gemini.temperature_default",
            default_value=0.7,
            type_converter=float
        )

        self.MAX_TOKENS_DEFAULT = self.create_persistent_config(
            env_name="GEMINI_MAX_TOKENS_DEFAULT",
            config_path="gemini.max_tokens_default",
            default_value=1000,
            type_converter=int
        )

        self.REQUEST_TIMEOUT = self.create_persistent_config(
            env_name="GEMINI_REQUEST_TIMEOUT",
            config_path="gemini.request_timeout",
            default_value=30,
            type_converter=int
        )

        return self.configs

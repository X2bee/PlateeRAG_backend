"""
Anthropic API 관련 설정
"""
from typing import Dict
from config.base_config import BaseConfig, PersistentConfig

class AnthropicConfig(BaseConfig):
    """Anthropic API 관련 설정 관리"""

    def initialize(self) -> Dict[str, PersistentConfig]:
        """Anthropic 관련 설정들을 초기화"""

        self.API_KEY = self.create_persistent_config(
            env_name="ANTHROPIC_API_KEY",
            config_path="anthropic.api_key",
            default_value="",
            file_path="anthropic_api_key.txt"
        )

        self.MODEL_DEFAULT = self.create_persistent_config(
            env_name="ANTHROPIC_MODEL_DEFAULT",
            config_path="anthropic.model_default",
            default_value="claude-sonnet-4-20250514"
        )

        self.API_BASE_URL = self.create_persistent_config(
            env_name="ANTHROPIC_API_BASE_URL",
            config_path="anthropic.api_base_url",
            default_value="https://api.anthropic.com"
        )

        self.TEMPERATURE_DEFAULT = self.create_persistent_config(
            env_name="ANTHROPIC_TEMPERATURE_DEFAULT",
            config_path="anthropic.temperature_default",
            default_value=0.7,
            type_converter=float
        )

        self.MAX_TOKENS_DEFAULT = self.create_persistent_config(
            env_name="ANTHROPIC_MAX_TOKENS_DEFAULT",
            config_path="anthropic.max_tokens_default",
            default_value=1000,
            type_converter=int
        )

        self.REQUEST_TIMEOUT = self.create_persistent_config(
            env_name="ANTHROPIC_REQUEST_TIMEOUT",
            config_path="anthropic.request_timeout",
            default_value=30,
            type_converter=int
        )

        return self.configs

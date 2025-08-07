"""
LLM 제공자 관련 설정
"""
from typing import Dict
from config.base_config import BaseConfig, PersistentConfig

class LLMConfig(BaseConfig):
    """LLM 제공자 관련 설정 관리"""

    def initialize(self) -> Dict[str, PersistentConfig]:
        """LLM 관련 설정들을 초기화"""

        # 기본 LLM 제공자 설정
        self.DEFAULT_PROVIDER = self.create_persistent_config(
            env_name="DEFAULT_LLM_PROVIDER",
            config_path="llm.default_provider",
            default_value="openai"
        )

        # LLM 자동 전환 설정
        self.AUTO_FALLBACK = self.create_persistent_config(
            env_name="LLM_AUTO_FALLBACK",
            config_path="llm.auto_fallback",
            default_value=True,
            type_converter=bool
        )

        # LLM 연결 테스트 타임아웃
        self.CONNECTION_TIMEOUT = self.create_persistent_config(
            env_name="LLM_CONNECTION_TIMEOUT",
            config_path="llm.connection_timeout",
            default_value=10,
            type_converter=int
        )

        # LLM 재시도 횟수
        self.MAX_RETRIES = self.create_persistent_config(
            env_name="LLM_MAX_RETRIES",
            config_path="llm.max_retries",
            default_value=3,
            type_converter=int
        )

        return self.configs

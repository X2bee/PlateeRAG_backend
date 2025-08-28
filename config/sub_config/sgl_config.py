"""
SGL API 관련 설정
"""
from typing import Dict
from config.base_config import BaseConfig, PersistentConfig

class SGLConfig(BaseConfig):
    """SGL API 관련 설정 관리"""

    def initialize(self) -> Dict[str, PersistentConfig]:
        """SGL 관련 설정들을 초기화"""

        # SGL API Base URL 설정
        self.SGL_API_BASE_URL = self.create_persistent_config(
            env_name="SGL_API_BASE_URL",
            config_path="SGL.api_base_url",
            default_value="http://localhost:12721/v1"
        )

        # SGL API Key (선택사항, 인증이 필요한 경우)
        self.SGL_API_KEY = self.create_persistent_config(
            env_name="SGL_API_KEY",
            config_path="SGL.api_key",
            default_value="",
            file_path="SGL_api_key.txt"
        )

        # 사용할 모델 이름
        self.SGL_MODEL_NAME = self.create_persistent_config(
            env_name="SGL_MODEL_NAME",
            config_path="SGL.model_name",
            default_value="Qwen/Qwen3-4B"
        )

        return self.configs

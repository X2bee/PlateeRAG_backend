"""
vLLM API 관련 설정
"""
from typing import Dict
from config.base_config import BaseConfig, PersistentConfig

class VLLMConfig(BaseConfig):
    """vLLM API 관련 설정 관리"""

    def initialize(self) -> Dict[str, PersistentConfig]:
        """vLLM 관련 설정들을 초기화"""

        # vLLM API Base URL 설정
        self.VLLM_API_BASE_URL = self.create_persistent_config(
            env_name="VLLM_API_BASE_URL",
            config_path="vllm.api_base_url",
            default_value="http://localhost:12721/v1"
        )

        # vLLM API Key (선택사항, 인증이 필요한 경우)
        self.VLLM_API_KEY = self.create_persistent_config(
            env_name="VLLM_API_KEY",
            config_path="vllm.api_key",
            default_value="",
            file_path="vllm_api_key.txt"
        )

        # 사용할 모델 이름
        self.VLLM_MODEL_NAME = self.create_persistent_config(
            env_name="VLLM_MODEL_NAME",
            config_path="vllm.model_name",
            default_value="Qwen/Qwen3-4B"
        )

        return self.configs

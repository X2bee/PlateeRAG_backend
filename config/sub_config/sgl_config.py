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
        self.API_BASE_URL = self.create_persistent_config(
            env_name="SGL_API_BASE_URL",
            config_path="SGL.api_base_url",
            default_value="http://localhost:12721/v1"
        )

        # SGL API Key (선택사항, 인증이 필요한 경우)
        self.API_KEY = self.create_persistent_config(
            env_name="SGL_API_KEY",
            config_path="SGL.api_key",
            default_value="",
            file_path="SGL_api_key.txt"
        )

        # 사용할 모델 이름
        self.MODEL_NAME = self.create_persistent_config(
            env_name="SGL_MODEL_NAME",
            config_path="SGL.model_name",
            default_value="Qwen/Qwen3-4B"
        )

        # 기본 온도 설정
        self.TEMPERATURE_DEFAULT = self.create_persistent_config(
            env_name="SGL_TEMPERATURE_DEFAULT",
            config_path="SGL.temperature_default",
            default_value=0.7,
            type_converter=float
        )

        # 기본 최대 토큰 설정
        self.MAX_TOKENS_DEFAULT = self.create_persistent_config(
            env_name="SGL_MAX_TOKENS_DEFAULT",
            config_path="SGL.max_tokens_default",
            default_value=512,
            type_converter=int
        )

        # Top-p (Nucleus Sampling)
        self.TOP_P = self.create_persistent_config(
            env_name="SGL_TOP_P",
            config_path="SGL.top_p",
            default_value=0.9,
            type_converter=float
        )

        # Frequency Penalty
        self.FREQUENCY_PENALTY = self.create_persistent_config(
            env_name="SGL_FREQUENCY_PENALTY",
            config_path="SGL.frequency_penalty",
            default_value=0.0,
            type_converter=float
        )

        # Presence Penalty
        self.PRESENCE_PENALTY = self.create_persistent_config(
            env_name="SGL_PRESENCE_PENALTY",
            config_path="SGL.presence_penalty",
            default_value=0.0,
            type_converter=float
        )

        # Random Seed
        self.SEED = self.create_persistent_config(
            env_name="SGL_SEED",
            config_path="SGL.seed",
            default_value=-1,  # -1은 랜덤 시드 사용
            type_converter=int
        )

        # API 요청 타임아웃
        self.REQUEST_TIMEOUT = self.create_persistent_config(
            env_name="SGL_TIMEOUT",
            config_path="SGL.request_timeout",
            default_value=60,
            type_converter=int
        )

        # 스트리밍 응답 사용 여부
        self.STREAM = self.create_persistent_config(
            env_name="SGL_STREAM",
            config_path="SGL.stream",
            default_value=False,
            type_converter=bool
        )

        # SGL 서버 연결 상태 확인 및 로깅
        if self.API_BASE_URL.value and self.API_BASE_URL.value.strip():
            self.logger.info(f"SGL server configured at: {self.API_BASE_URL.value}")
            
            # API 키가 설정된 경우 환경변수에 설정
            if self.API_KEY.value and self.API_KEY.value.strip():
                import os
                os.environ["SGL_API_KEY"] = self.API_KEY.value.strip()
                self.logger.info("SGL API key set in environment")
        else:
            self.logger.warning("SGL server URL not configured")

        # 모델 이름 확인
        if self.MODEL_NAME.value and self.MODEL_NAME.value.strip():
            self.logger.info(f"SGL model configured: {self.MODEL_NAME.value}")
        else:
            self.logger.warning("SGL model name not configured")
        
        return self.configs
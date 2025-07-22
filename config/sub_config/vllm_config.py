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
        self.API_BASE_URL = self.create_persistent_config(
            env_name="VLLM_API_BASE_URL",
            config_path="vllm.api_base_url",
            default_value="http://localhost:8000/v1"
        )

        # vLLM API Key (선택사항, 인증이 필요한 경우)
        self.API_KEY = self.create_persistent_config(
            env_name="VLLM_API_KEY",
            config_path="vllm.api_key",
            default_value="",
            file_path="vllm_api_key.txt"
        )

        # 사용할 모델 이름
        self.MODEL_NAME = self.create_persistent_config(
            env_name="VLLM_MODEL_NAME",
            config_path="vllm.model_name",
            default_value="meta-llama/Llama-2-7b-chat-hf"
        )

        # 기본 온도 설정
        self.TEMPERATURE_DEFAULT = self.create_persistent_config(
            env_name="VLLM_TEMPERATURE_DEFAULT",
            config_path="vllm.temperature_default",
            default_value=0.7,
            type_converter=float
        )

        # 기본 최대 토큰 설정
        self.MAX_TOKENS_DEFAULT = self.create_persistent_config(
            env_name="VLLM_MAX_TOKENS_DEFAULT",
            config_path="vllm.max_tokens_default",
            default_value=512,
            type_converter=int
        )

        # Top-p (Nucleus Sampling)
        self.TOP_P = self.create_persistent_config(
            env_name="VLLM_TOP_P",
            config_path="vllm.top_p",
            default_value=0.9,
            type_converter=float
        )

        # Top-k
        self.TOP_K = self.create_persistent_config(
            env_name="VLLM_TOP_K",
            config_path="vllm.top_k",
            default_value=-1,  # -1은 비활성화를 의미
            type_converter=int
        )

        # Frequency Penalty
        self.FREQUENCY_PENALTY = self.create_persistent_config(
            env_name="VLLM_FREQUENCY_PENALTY",
            config_path="vllm.frequency_penalty",
            default_value=0.0,
            type_converter=float
        )

        # Presence Penalty
        self.PRESENCE_PENALTY = self.create_persistent_config(
            env_name="VLLM_PRESENCE_PENALTY",
            config_path="vllm.presence_penalty",
            default_value=0.0,
            type_converter=float
        )

        # Repetition Penalty
        self.REPETITION_PENALTY = self.create_persistent_config(
            env_name="VLLM_REPETITION_PENALTY",
            config_path="vllm.repetition_penalty",
            default_value=1.0,
            type_converter=float
        )

        # Best of (여러 생성 결과 중 최고 선택)
        self.BEST_OF = self.create_persistent_config(
            env_name="VLLM_BEST_OF",
            config_path="vllm.best_of",
            default_value=1,
            type_converter=int
        )

        # Beam Search 사용 여부
        self.USE_BEAM_SEARCH = self.create_persistent_config(
            env_name="VLLM_USE_BEAM_SEARCH",
            config_path="vllm.use_beam_search",
            default_value=False,
            type_converter=bool
        )

        # Stop Sequences (JSON 문자열로 저장)
        self.STOP_SEQUENCES = self.create_persistent_config(
            env_name="VLLM_STOP_SEQUENCES",
            config_path="vllm.stop_sequences",
            default_value='["</s>", "Human:", "Assistant:"]'
        )

        # Random Seed
        self.SEED = self.create_persistent_config(
            env_name="VLLM_SEED",
            config_path="vllm.seed",
            default_value=-1,  # -1은 랜덤 시드 사용
            type_converter=int
        )

        # API 요청 타임아웃
        self.REQUEST_TIMEOUT = self.create_persistent_config(
            env_name="VLLM_TIMEOUT",
            config_path="vllm.request_timeout",
            default_value=60,
            type_converter=int
        )

        # 스트리밍 응답 사용 여부
        self.STREAM = self.create_persistent_config(
            env_name="VLLM_STREAM",
            config_path="vllm.stream",
            default_value=False,
            type_converter=bool
        )

        # Log Probabilities
        self.LOGPROBS = self.create_persistent_config(
            env_name="VLLM_LOGPROBS",
            config_path="vllm.logprobs",
            default_value=0,  # 0은 비활성화
            type_converter=int
        )

        # Echo Input (입력 프롬프트를 출력에 포함)
        self.ECHO = self.create_persistent_config(
            env_name="VLLM_ECHO",
            config_path="vllm.echo",
            default_value=False,
            type_converter=bool
        )

        # Organization ID (선택사항)
        self.ORGANIZATION_ID = self.create_persistent_config(
            env_name="VLLM_ORGANIZATION_ID",
            config_path="vllm.organization_id",
            default_value=""
        )

        # vLLM 서버 연결 상태 확인 및 로깅
        if self.API_BASE_URL.value and self.API_BASE_URL.value.strip():
            self.logger.info(f"vLLM server configured at: {self.API_BASE_URL.value}")
            
            # API 키가 설정된 경우 환경변수에 설정
            if self.API_KEY.value and self.API_KEY.value.strip():
                import os
                os.environ["VLLM_API_KEY"] = self.API_KEY.value.strip()
                self.logger.info("vLLM API key set in environment")
        else:
            self.logger.warning("vLLM server URL not configured")

        # 모델 이름 확인
        if self.MODEL_NAME.value and self.MODEL_NAME.value.strip():
            self.logger.info(f"vLLM model configured: {self.MODEL_NAME.value}")
        else:
            self.logger.warning("vLLM model name not configured")
        
        return self.configs
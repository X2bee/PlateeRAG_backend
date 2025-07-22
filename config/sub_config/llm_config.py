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

        # 기본 LLM 제공자 검증
        valid_providers = ["openai", "vllm"]
        if self.DEFAULT_PROVIDER.value not in valid_providers:
            self.logger.warning(
                f"Invalid LLM provider '{self.DEFAULT_PROVIDER.value}'. "
                f"Valid options: {', '.join(valid_providers)}. "
                f"Defaulting to 'openai'."
            )
            self.DEFAULT_PROVIDER.update_value("openai")

        # 환경변수에 기본 제공자 설정
        import os
        os.environ["DEFAULT_LLM_PROVIDER"] = self.DEFAULT_PROVIDER.value
        self.logger.info(f"Default LLM provider set to: {self.DEFAULT_PROVIDER.value}")
        
        return self.configs

    def get_current_provider(self) -> str:
        """현재 설정된 기본 LLM 제공자 반환"""
        return self.DEFAULT_PROVIDER.value

    def set_default_provider(self, provider: str) -> bool:
        """기본 LLM 제공자 변경"""
        valid_providers = ["openai", "vllm"]
        
        if provider not in valid_providers:
            self.logger.error(f"Invalid provider '{provider}'. Valid options: {', '.join(valid_providers)}")
            return False
        
        try:
            self.DEFAULT_PROVIDER.update_value(provider)
            
            # 환경변수도 업데이트
            import os
            os.environ["DEFAULT_LLM_PROVIDER"] = provider
            
            self.logger.info(f"Default LLM provider changed to: {provider}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to set default provider to '{provider}': {e}")
            return False

    def is_provider_available(self, provider: str) -> bool:
        """특정 LLM 제공자가 사용 가능한지 확인"""
        if provider == "openai":
            # OpenAI 설정 확인
            from config.openai_config import OpenAIConfig
            openai_config = OpenAIConfig()
            return bool(openai_config.API_KEY.value and openai_config.API_KEY.value.strip())
        
        elif provider == "vllm":
            # vLLM 설정 확인
            from config.vllm_config import VLLMConfig
            vllm_config = VLLMConfig()
            return bool(vllm_config.API_BASE_URL.value and vllm_config.API_BASE_URL.value.strip())
        
        return False

    def get_available_providers(self) -> list:
        """사용 가능한 LLM 제공자 목록 반환"""
        providers = []
        
        if self.is_provider_available("openai"):
            providers.append("openai")
        
        if self.is_provider_available("vllm"):
            providers.append("vllm")
        
        return providers

    def auto_select_provider(self) -> str:
        """사용 가능한 제공자 중에서 자동으로 선택"""
        available_providers = self.get_available_providers()
        
        if not available_providers:
            self.logger.warning("No LLM providers available")
            return self.DEFAULT_PROVIDER.value
        
        # 현재 설정된 제공자가 사용 가능하면 그대로 유지
        if self.DEFAULT_PROVIDER.value in available_providers:
            return self.DEFAULT_PROVIDER.value
        
        # 사용 가능한 첫 번째 제공자로 자동 전환
        new_provider = available_providers[0]
        
        if self.AUTO_FALLBACK.value:
            self.logger.info(f"Auto-switching to available provider: {new_provider}")
            self.set_default_provider(new_provider)
            return new_provider
        else:
            self.logger.warning(f"Current provider '{self.DEFAULT_PROVIDER.value}' not available. Auto-fallback disabled.")
            return self.DEFAULT_PROVIDER.value

    def validate_configuration(self) -> Dict[str, any]:
        """LLM 설정 유효성 검사"""
        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "providers": {}
        }
        
        # 각 제공자별 상태 확인
        for provider in ["openai", "vllm"]:
            is_available = self.is_provider_available(provider)
            result["providers"][provider] = {
                "available": is_available,
                "configured": is_available  # 현재는 available과 동일하게 처리
            }
        
        # 사용 가능한 제공자가 있는지 확인
        available_providers = self.get_available_providers()
        if not available_providers:
            result["valid"] = False
            result["errors"].append("No LLM providers are properly configured")
        
        # 현재 기본 제공자가 사용 가능한지 확인
        if self.DEFAULT_PROVIDER.value not in available_providers:
            result["warnings"].append(
                f"Default provider '{self.DEFAULT_PROVIDER.value}' is not available. "
                f"Available providers: {', '.join(available_providers) if available_providers else 'None'}"
            )
        
        return result
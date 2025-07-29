"""
Collection 관리 관련 설정
"""
from typing import Dict
from config.base_config import BaseConfig, PersistentConfig

class CollectionConfig(BaseConfig):
    """Collection 관리 관련 설정 관리"""

    def initialize(self) -> Dict[str, PersistentConfig]:
        """Collection 관련 설정들을 초기화"""

        # 이미지-텍스트 모델 제공자 설정
        self.IMAGE_TEXT_MODEL_PROVIDER = self.create_persistent_config(
            env_name="IMAGE_TEXT_MODEL_PROVIDER",
            config_path="collection.image_text_model_provider",
            default_value="openai"
        )

        # 이미지-텍스트 모델 Base URL
        self.IMAGE_TEXT_BASE_URL = self.create_persistent_config(
            env_name="IMAGE_TEXT_BASE_URL",
            config_path="collection.image_text_base_url",
            default_value="https://api.openai.com/v1"
        )

        # 이미지-텍스트 모델 API 키
        self.IMAGE_TEXT_API_KEY = self.create_persistent_config(
            env_name="IMAGE_TEXT_API_KEY",
            config_path="collection.image_text_api_key",
            default_value=""
        )

        # 이미지-텍스트 모델 이름
        self.IMAGE_TEXT_MODEL_NAME = self.create_persistent_config(
            env_name="IMAGE_TEXT_MODEL_NAME",
            config_path="collection.image_text_model_name",
            default_value="gpt-4-vision-preview"
        )

        # 온도 설정
        self.IMAGE_TEXT_TEMPERATURE = self.create_persistent_config(
            env_name="IMAGE_TEXT_TEMPERATURE",
            config_path="collection.image_text_temperature",
            default_value=0.7,
            type_converter=float
        )

        # 이미지 품질 설정
        self.IMAGE_QUALITY = self.create_persistent_config(
            env_name="IMAGE_QUALITY",
            config_path="collection.image_quality",
            default_value="auto"
        )

        # 제공자 검증
        valid_providers = ["openai", "vLLM", "no_model"]
        if self.IMAGE_TEXT_MODEL_PROVIDER.value not in valid_providers:
            self.logger.warning(
                f"Invalid image-text model provider '{self.IMAGE_TEXT_MODEL_PROVIDER.value}'. "
                f"Valid options: {', '.join(valid_providers)}. "
                f"Defaulting to 'openai'."
            )
            # 올바른 방법으로 값 설정
            self.IMAGE_TEXT_MODEL_PROVIDER.value = "openai"
            self.IMAGE_TEXT_MODEL_PROVIDER.save()

        # 환경변수에 설정 적용
        import os
        os.environ["IMAGE_TEXT_MODEL_PROVIDER"] = self.IMAGE_TEXT_MODEL_PROVIDER.value
        self.logger.info(f"Image-text model provider set to: {self.IMAGE_TEXT_MODEL_PROVIDER.value}")
        
        return self.configs

    def get_current_provider(self) -> str:
        """현재 설정된 이미지-텍스트 모델 제공자 반환"""
        return self.IMAGE_TEXT_MODEL_PROVIDER.value

    def set_provider(self, provider: str) -> bool:
        """이미지-텍스트 모델 제공자 변경"""
        valid_providers = ["openai", "vLLM", "no_model"]
        
        if provider not in valid_providers:
            self.logger.error(f"Invalid provider '{provider}'. Valid options: {', '.join(valid_providers)}")
            return False
        
        try:
            # 올바른 방법으로 값 설정
            self.IMAGE_TEXT_MODEL_PROVIDER.value = provider
            self.IMAGE_TEXT_MODEL_PROVIDER.save()
            
            # 환경변수도 업데이트
            import os
            os.environ["IMAGE_TEXT_MODEL_PROVIDER"] = provider
            
            self.logger.info(f"Image-text model provider changed to: {provider}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to set provider to '{provider}': {e}")
            return False

    def is_provider_configured(self, provider: str = None) -> bool:
        """특정 이미지-텍스트 모델 제공자가 구성되어 있는지 확인"""
        if provider is None:
            provider = self.IMAGE_TEXT_MODEL_PROVIDER.value
        
        if provider == "no_model":
            return True  # no_model은 항상 사용 가능
        
        # 공통 요구사항: API 키와 모델 이름
        if not (self.IMAGE_TEXT_API_KEY.value and self.IMAGE_TEXT_API_KEY.value.strip()):
            return False
        
        if not (self.IMAGE_TEXT_MODEL_NAME.value and self.IMAGE_TEXT_MODEL_NAME.value.strip()):
            return False
        
        # Base URL 확인
        if not (self.IMAGE_TEXT_BASE_URL.value and self.IMAGE_TEXT_BASE_URL.value.strip()):
            return False
        
        return True

    def get_provider_config(self, provider: str = None) -> Dict[str, any]:
        """특정 제공자의 설정 정보 반환"""
        if provider is None:
            provider = self.IMAGE_TEXT_MODEL_PROVIDER.value
        
        config = {
            "provider": provider,
            "base_url": self.IMAGE_TEXT_BASE_URL.value,
            "api_key_configured": bool(self.IMAGE_TEXT_API_KEY.value and self.IMAGE_TEXT_API_KEY.value.strip()),
            "model_name": self.IMAGE_TEXT_MODEL_NAME.value,
            "temperature": self.IMAGE_TEXT_TEMPERATURE.value,
            "image_quality": self.IMAGE_QUALITY.value,
            "configured": self.is_provider_configured(provider)
        }
        
        return config

    def update_provider_config(self, **kwargs) -> bool:
        """제공자 설정 업데이트"""
        try:
            if "base_url" in kwargs:
                self.IMAGE_TEXT_BASE_URL.value = kwargs["base_url"]
                self.IMAGE_TEXT_BASE_URL.save()
            
            if "api_key" in kwargs:
                self.IMAGE_TEXT_API_KEY.value = kwargs["api_key"]
                self.IMAGE_TEXT_API_KEY.save()
            
            if "model_name" in kwargs:
                self.IMAGE_TEXT_MODEL_NAME.value = kwargs["model_name"]
                self.IMAGE_TEXT_MODEL_NAME.save()
            
            if "temperature" in kwargs:
                temp = float(kwargs["temperature"])
                if 0.0 <= temp <= 2.0:
                    self.IMAGE_TEXT_TEMPERATURE.value = temp
                    self.IMAGE_TEXT_TEMPERATURE.save()
                else:
                    raise ValueError("Temperature must be between 0.0 and 2.0")
            
            if "image_quality" in kwargs:
                quality = kwargs["image_quality"]
                if quality in ["auto", "low", "high"]:
                    self.IMAGE_QUALITY.value = quality
                    self.IMAGE_QUALITY.save()
                else:
                    raise ValueError("Image quality must be 'auto', 'low', or 'high'")
            
            self.logger.info("Collection provider config updated successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update provider config: {e}")
            return False

    def validate_configuration(self) -> Dict[str, any]:
        """Collection 설정 유효성 검사"""
        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "config": {}
        }
        
        # 현재 제공자 설정 확인
        current_provider = self.get_current_provider()
        provider_config = self.get_provider_config(current_provider)
        result["config"] = provider_config
        
        # 필수 설정 확인
        if not provider_config["configured"]:
            result["valid"] = False
            result["errors"].append("Image-text model provider is not properly configured")
        
        if current_provider != "no_model":
            if not provider_config["api_key_configured"]:
                result["errors"].append("API key is required")
            
            if not provider_config["model_name"]:
                result["errors"].append("Model name is required")
            
            if not provider_config["base_url"]:
                result["errors"].append("Base URL is required")
        
        # 온도 값 검증
        if not (0.0 <= self.IMAGE_TEXT_TEMPERATURE.value <= 2.0):
            result["warnings"].append("Temperature should be between 0.0 and 2.0")
        
        return result

    def test_connection(self) -> Dict[str, any]:
        """이미지-텍스트 모델 연결 테스트"""
        result = {
            "provider": self.get_current_provider(),
            "success": False,
            "message": "",
            "response_time": None
        }
        
        if not self.is_provider_configured():
            result["message"] = "Provider is not properly configured"
            return result
        
        try:
            import time
            import requests
            
            start_time = time.time()
            
            # 간단한 연결 테스트 (실제 구현에서는 각 제공자별로 다른 엔드포인트 사용)
            headers = {
                "Authorization": f"Bearer {self.IMAGE_TEXT_API_KEY.value.strip()}",
                "Content-Type": "application/json"
            }
            
            # 모델 목록 조회로 연결 테스트
            test_url = f"{self.IMAGE_TEXT_BASE_URL.value.rstrip('/')}/models"
            
            response = requests.get(
                test_url,
                headers=headers,
                timeout=10
            )
            
            result["response_time"] = time.time() - start_time
            
            if response.status_code == 200:
                result["success"] = True
                result["message"] = "Connection test successful"
            else:
                result["message"] = f"Server responded with status {response.status_code}"
                
        except Exception as e:
            result["message"] = f"Connection test failed: {str(e)}"
            self.logger.error(f"Collection connection test failed: {e}")
        
        return result

    def get_supported_formats(self) -> list:
        """지원되는 이미지 형식 반환"""
        return ["jpg", "jpeg", "png", "gif", "webp"]

    def get_max_image_size(self) -> int:
        """최대 이미지 크기 반환 (MB)"""
        return 10  # 기본값 10MB

    def get_model_info(self) -> Dict[str, any]:
        """현재 모델 정보 반환"""
        return {
            "provider": self.get_current_provider(),
            "model_name": self.IMAGE_TEXT_MODEL_NAME.value,
            "temperature": self.IMAGE_TEXT_TEMPERATURE.value,
            "image_quality": self.IMAGE_QUALITY.value,
            "supported_formats": self.get_supported_formats(),
            "max_image_size_mb": self.get_max_image_size()
        }
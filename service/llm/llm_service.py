"""
LLM 제공자 관리 서비스
"""
import logging
from typing import Dict, Any, Optional
from service.llm.openai_service import OpenAIService
from service.llm.vllm_service import VLLMService
from service.llm.sgl_service import SGLService
from service.llm.anthropic_service import AnthropicService
from service.llm.gemini_service import GeminiService

logger = logging.getLogger(__name__)

class LLMService:
    """LLM 제공자 관리 서비스"""

    def __init__(self):
        self.openai_service = OpenAIService()
        self.vllm_service = VLLMService()
        self.sgl_service = SGLService()
        self.anthropic_service = AnthropicService()
        self.gemini_service = GeminiService()

    def get_llm_client(self, provider: str):
        """지정된 제공자의 LLM 클라이언트 반환"""
        if provider == 'openai':
            return self.openai_service
        elif provider == 'vllm':
            return self.vllm_service
        elif provider == 'sgl':
            return self.sgl_service
        elif provider == 'anthropic':
            return self.anthropic_service
        elif provider == 'gemini':
            return self.gemini_service
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    async def test_provider_connection(self, provider: str, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """제공자별 연결 테스트"""
        try:
            if provider == 'openai':
                return await self.test_openai_connection(config_data)
            elif provider == 'vllm':
                return await self.test_vllm_connection(config_data)
            elif provider == 'sgl':
                return await self.test_sgl_connection(config_data)
            elif provider == 'anthropic':
                return await self.test_anthropic_connection(config_data)
            elif provider == 'gemini':
                return await self.test_gemini_connection(config_data)
            else:
                return {
                    "status": "error",
                    "message": f"Unsupported LLM provider: {provider}"
                }
        except Exception as e:
            logger.error(f"Error testing {provider} connection: {e}")
            return {
                "status": "error",
                "message": f"Connection test failed: {str(e)}"
            }

    async def test_openai_connection(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """OpenAI 연결 테스트"""
        api_key = config_data.get('api_key')
        base_url = config_data.get('base_url', 'https://api.openai.com/v1')
        model = config_data.get('model', 'gpt-3.5-turbo')

        return await self.openai_service.test_connection(api_key, base_url, model)

    async def test_vllm_connection(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """vLLM 연결 테스트"""
        base_url = config_data.get('base_url')
        api_key = config_data.get('api_key')
        model_name = config_data.get('model_name')

        return await self.vllm_service.test_connection(base_url, api_key, model_name)

    async def test_sgl_connection(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """SGL 연결 테스트"""
        base_url = config_data.get('base_url')
        api_key = config_data.get('api_key')
        model_name = config_data.get('model_name')

        return await self.sgl_service.test_connection(base_url, api_key, model_name)

    async def test_anthropic_connection(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Anthropic 연결 테스트"""
        base_url = config_data.get('base_url')
        api_key = config_data.get('api_key')
        model_name = config_data.get('model_name')

        return await self.anthropic_service.test_connection(api_key, base_url, model_name)

    async def test_gemini_connection(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Gemini 연결 테스트"""
        base_url = config_data.get('base_url')
        api_key = config_data.get('api_key')
        model_name = config_data.get('model_name')

        return await self.gemini_service.test_connection(api_key, base_url, model_name)


    def validate_provider_config(self, provider: str, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """제공자별 설정 유효성 검사"""
        if provider == 'openai':
            return self._validate_openai_config(config_data)
        elif provider == 'vllm':
            return self._validate_vllm_config(config_data)
        elif provider == 'sgl':
            return self._validate_sgl_config(config_data)
        elif provider == 'anthropic':
            return self._validate_anthropic_config(config_data)
        elif provider == 'gemini':
            return self._validate_gemini_config(config_data)
        else:
            return {"valid": False, "error": f"Unsupported provider: {provider}"}

    def _validate_openai_config(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """OpenAI 설정 유효성 검사"""
        api_key = config_data.get('api_key')

        if not api_key or not api_key.strip():
            return {"valid": False, "error": "OpenAI API Key is required"}

        return {"valid": True}

    def _validate_vllm_config(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """vLLM 설정 유효성 검사"""
        base_url = config_data.get('base_url')

        if not base_url or not base_url.strip():
            return {"valid": False, "error": "vLLM Base URL is required"}

        return {"valid": True}

    def _validate_sgl_config(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """SGL 설정 유효성 검사"""
        base_url = config_data.get('base_url')
        model_name = config_data.get('model_name')

        errors = []
        warnings = []

        if not base_url or not base_url.strip():
            errors.append("SGL Base URL is required")

        if not model_name or not model_name.strip():
            errors.append("SGL Model Name is required")
        else:
            # 모델 이름 형식 검증
            model_validation = self.sgl_service.validate_model_name(model_name)
            if not model_validation.get("valid"):
                if "error" in model_validation:
                    errors.append(model_validation["error"])
                if "warning" in model_validation:
                    warnings.append(model_validation["warning"])

        result = {"valid": len(errors) == 0}
        if errors:
            result["errors"] = errors
        if warnings:
            result["warnings"] = warnings

        return result

    def _validate_anthropic_config(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Anthropic 설정 유효성 검사"""
        api_key = config_data.get('api_key')
        if not api_key or not api_key.strip():
            return {"valid": False, "error": "Anthropic API Key is required"}
        return {"valid": True}

    def _validate_gemini_config(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Gemini 설정 유효성 검사"""
        api_key = config_data.get('api_key')
        if not api_key or not api_key.strip():
            return {"valid": False, "error": "Gemini API Key is required"}
        return {"valid": True}


    async def get_provider_models(self, provider: str, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """제공자별 사용 가능한 모델 목록 조회"""
        try:
            if provider == 'sgl':
                base_url = config_data.get('base_url')
                api_key = config_data.get('api_key')
                return await self.sgl_service.get_models(base_url, api_key)
            elif provider == 'vllm':
                # vLLM의 경우 기존 연결 테스트에서 모델 정보를 가져올 수 있음
                connection_result = await self.test_vllm_connection(config_data)
                if connection_result.get("status") == "success":
                    return {
                        "status": "success",
                        "models": [{"id": model} for model in connection_result.get("available_models", [])],
                        "count": connection_result.get("models_count", 0)
                    }
                else:
                    return {"status": "error", "message": "Failed to connect to vLLM server"}
            else:
                return {"status": "error", "message": f"Model listing not supported for provider: {provider}"}

        except Exception as e:
            logger.error(f"Error getting models for {provider}: {e}")
            return {"status": "error", "message": f"Failed to get models: {str(e)}"}

    def get_supported_providers(self) -> list:
        """지원되는 제공자 목록 반환"""
        return ['openai', 'vllm', 'sgl', 'anthropic', 'gemini']

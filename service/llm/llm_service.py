"""
LLM 제공자 관리 서비스
"""
import logging
from typing import Dict, Any, Optional
from service.llm.openai_service import OpenAIService
from service.llm.vllm_service import VLLMService

logger = logging.getLogger(__name__)

class LLMService:
    """LLM 제공자 관리 서비스"""
    
    def __init__(self):
        self.openai_service = OpenAIService()
        self.vllm_service = VLLMService()
    
    def get_llm_client(self, provider: str):
        """지정된 제공자의 LLM 클라이언트 반환"""
        if provider == 'openai':
            return self.openai_service
        elif provider == 'vllm':
            return self.vllm_service
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    
    def test_provider_connection(self, provider: str, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """제공자별 연결 테스트"""
        if provider == 'openai':
            return self.test_openai_connection(config_data)
        elif provider == 'vllm':
            return self.test_vllm_connection(config_data)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    
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
    
    def validate_provider_config(self, provider: str, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """제공자별 설정 유효성 검사"""
        if provider == 'openai':
            return self._validate_openai_config(config_data)
        elif provider == 'vllm':
            return self._validate_vllm_config(config_data)
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
"""
vLLM API 서비스
"""
import requests
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class VLLMService:
    """vLLM API 서비스 클래스"""
    
    def __init__(self):
        """생성자 - 설정은 메서드 호출 시 전달받음"""
        pass
    
    async def test_connection(self, base_url: str, api_key: str = None, model_name: str = None) -> Dict[str, Any]:
        """vLLM 서버 연결 테스트"""
        if not base_url or not base_url.strip():
            raise ValueError("vLLM Base URL is not configured")
        
        base_url = base_url.rstrip('/')
        headers = {"Content-Type": "application/json"}
        
        if api_key and api_key.strip():
            headers["Authorization"] = f"Bearer {api_key.strip()}"
        
        # 1. Health check 또는 models 엔드포인트로 연결 테스트
        try:
            # 먼저 models 엔드포인트 시도
            models_url = f"{base_url}/models"
            response = requests.get(models_url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                models_data = response.json()
                available_models = [model.get("id", "") for model in models_data.get("data", [])]
                
                # 간단한 completion 테스트 (모델이 있는 경우)
                completion_test = None
                if model_name and model_name in available_models:
                    completion_test = await self.test_completion(base_url, api_key, model_name)
                
                return {
                    "status": "success",
                    "message": "vLLM server connection successful",
                    "api_url": base_url,
                    "models_count": len(available_models),
                    "available_models": available_models,
                    "configured_model": model_name,
                    "model_available": model_name in available_models if model_name else None,
                    "completion_test": completion_test
                }
            elif response.status_code == 401:
                raise ValueError("Invalid vLLM API key")
            else:
                # Health check 시도
                health_url = f"{base_url}/health"
                health_response = requests.get(health_url, headers=headers, timeout=10)
                
                if health_response.status_code == 200:
                    return {
                        "status": "success",
                        "message": "vLLM server is healthy",
                        "api_url": base_url,
                        "configured_model": model_name
                    }
                else:
                    raise requests.exceptions.HTTPError(f"HTTP {response.status_code}: {response.text}")
                
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to connect to vLLM server at {base_url}: {str(e)}")
    
    async def test_completion(self, base_url: str, api_key: str, model_name: str) -> Dict[str, Any]:
        """vLLM completion 간단 테스트"""
        try:
            headers = {"Content-Type": "application/json"}
            if api_key and api_key.strip():
                headers["Authorization"] = f"Bearer {api_key.strip()}"
            
            completion_url = f"{base_url}/chat/completions"
            
            test_payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 5,
                "temperature": 0
            }
            
            response = requests.post(completion_url, headers=headers, json=test_payload, timeout=15)
            
            if response.status_code == 200:
                return {"status": "success", "message": "Completion test passed"}
            else:
                return {"status": "warning", "message": f"Completion test failed: HTTP {response.status_code}"}
                
        except Exception as e:
            return {"status": "warning", "message": f"Completion test failed: {str(e)}"}
"""
SGL API 서비스
"""
import requests
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class SGLService:
    """SGL API 서비스 클래스"""
    
    def __init__(self):
        """생성자 - 설정은 메서드 호출 시 전달받음"""
        pass
    
    async def test_connection(self, base_url: str, api_key: str = None, model_name: str = None) -> Dict[str, Any]:
        """SGL 서버 연결 테스트"""
        if not base_url or not base_url.strip():
            raise ValueError("SGL Base URL is not configured")
        
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
                elif model_name:
                    # 모델 목록에 없어도 설정된 모델로 테스트 시도
                    completion_test = await self.test_completion(base_url, api_key, model_name)
                
                return {
                    "status": "success",
                    "message": "SGL server connection successful",
                    "api_url": base_url,
                    "models_count": len(available_models),
                    "available_models": available_models,
                    "configured_model": model_name,
                    "model_available": model_name in available_models if model_name and available_models else None,
                    "completion_test": completion_test
                }
            elif response.status_code == 401:
                raise ValueError("Invalid SGL API key")
            elif response.status_code == 404:
                # /models 엔드포인트가 없는 경우, 직접 completion 테스트
                if model_name:
                    completion_test = await self.test_completion(base_url, api_key, model_name)
                    if completion_test.get("status") == "success":
                        return {
                            "status": "success",
                            "message": "SGL server connection successful (direct completion test)",
                            "api_url": base_url,
                            "configured_model": model_name,
                            "completion_test": completion_test
                        }
                    else:
                        raise ConnectionError("SGL server connection failed - no /models endpoint and completion test failed")
                else:
                    raise ConnectionError("SGL server connection failed - no /models endpoint and no model configured for testing")
            else:
                # Health check 시도 (SGL이 health 엔드포인트를 지원하는 경우)
                try:
                    health_url = f"{base_url}/health"
                    health_response = requests.get(health_url, headers=headers, timeout=10)
                    
                    if health_response.status_code == 200:
                        return {
                            "status": "success",
                            "message": "SGL server is healthy",
                            "api_url": base_url,
                            "configured_model": model_name
                        }
                    else:
                        raise requests.exceptions.HTTPError(f"HTTP {response.status_code}: {response.text}")
                except requests.exceptions.RequestException:
                    # Health check도 실패하면 원래 에러 반환
                    raise requests.exceptions.HTTPError(f"HTTP {response.status_code}: {response.text}")
                
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to connect to SGL server at {base_url}: {str(e)}")
    
    async def test_completion(self, base_url: str, api_key: str, model_name: str) -> Dict[str, Any]:
        """SGL completion 간단 테스트"""
        try:
            headers = {"Content-Type": "application/json"}
            if api_key and api_key.strip():
                headers["Authorization"] = f"Bearer {api_key.strip()}"
            
            # SGL은 OpenAI 호환 API를 사용
            completion_url = f"{base_url}/chat/completions"
            
            test_payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 5,
                "temperature": 0
            }
            
            response = requests.post(completion_url, headers=headers, json=test_payload, timeout=15)
            
            if response.status_code == 200:
                result_data = response.json()
                content = ""
                
                # 응답에서 실제 내용 추출
                if "choices" in result_data and len(result_data["choices"]) > 0:
                    choice = result_data["choices"][0]
                    if "message" in choice and "content" in choice["message"]:
                        content = choice["message"]["content"]
                
                return {
                    "status": "success", 
                    "message": "Completion test passed",
                    "response_content": content.strip() if content else "No content",
                    "model_used": result_data.get("model", model_name)
                }
            else:
                error_msg = "Unknown error"
                try:
                    error_data = response.json()
                    error_msg = error_data.get("error", {}).get("message", error_msg)
                except:
                    error_msg = response.text[:200] if response.text else f"HTTP {response.status_code}"
                
                return {
                    "status": "warning", 
                    "message": f"Completion test failed: {error_msg}"
                }
                
        except Exception as e:
            return {
                "status": "warning", 
                "message": f"Completion test failed: {str(e)}"
            }
    
    async def get_models(self, base_url: str, api_key: str = None) -> Dict[str, Any]:
        """SGL 서버에서 사용 가능한 모델 목록 조회"""
        try:
            headers = {"Content-Type": "application/json"}
            if api_key and api_key.strip():
                headers["Authorization"] = f"Bearer {api_key.strip()}"
            
            models_url = f"{base_url.rstrip('/')}/models"
            response = requests.get(models_url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                models_data = response.json()
                models = []
                
                for model in models_data.get("data", []):
                    models.append({
                        "id": model.get("id", ""),
                        "object": model.get("object", "model"),
                        "created": model.get("created"),
                        "owned_by": model.get("owned_by", "sgl")
                    })
                
                return {
                    "status": "success",
                    "models": models,
                    "count": len(models)
                }
            else:
                return {
                    "status": "error",
                    "message": f"Failed to get models: HTTP {response.status_code}"
                }
                
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to get models: {str(e)}"
            }
    
    def validate_model_name(self, model_name: str) -> Dict[str, Any]:
        """모델 이름 유효성 검사"""
        if not model_name or not model_name.strip():
            return {
                "valid": False,
                "error": "Model name is required"
            }
        
        # 기본적인 모델 이름 패턴 검사
        if "/" not in model_name:
            return {
                "valid": False,
                "warning": "Model name should typically include organization/model format (e.g., 'Qwen/Qwen3-4B')"
            }
        
        return {"valid": True}
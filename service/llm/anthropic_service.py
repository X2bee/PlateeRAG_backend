"""
OpenAI API 서비스
"""
import requests
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class AnthropicService:
    """Anthropic API 서비스 클래스"""

    def __init__(self):
        """생성자 - 설정은 메서드 호출 시 전달받음"""
        pass

    async def test_connection(self, api_key: str, base_url: str = "https://api.anthropic.com", model: str = "claude-sonnet-4-20250514") -> Dict[str, Any]:
        """Anthropic API 연결 테스트"""
        if not api_key or not api_key.strip():
            raise ValueError("Anthropic API Key is not configured")

        api_key = api_key.strip()
        base_url = base_url.rstrip('/')
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        # Models 엔드포인트로 연결 테스트
        models_url = f"{base_url}/v1/models"

        try:
            response = requests.get(models_url, headers=headers, timeout=10)

            if response.status_code == 200:
                models_data = response.json()
                available_models = [model.get("id", "") for model in models_data.get("data", [])]

                completion_test = await self.test_completion(api_key, base_url, model)

                return {
                    "status": "success",
                    "message": "Anthropic API connection successful",
                    "api_url": base_url,
                    "models_count": len(available_models),
                    "configured_model": model,
                    "model_available": model in available_models if model else None,
                    "completion_test": completion_test
                }
            elif response.status_code == 401:
                raise ValueError("Invalid Anthropic API key")
            else:
                raise requests.exceptions.HTTPError(f"HTTP {response.status_code}: {response.text}")

        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to connect to Anthropic API at {base_url}: {str(e)}")

    async def test_completion(self, api_key: str, base_url: str, model: str) -> Dict[str, Any]:
        """Anthropic completion 간단 테스트"""
        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            completion_url = f"{base_url}/v1/messages"

            test_payload = {
                "model": model,
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

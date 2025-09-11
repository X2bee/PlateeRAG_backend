"""
OpenAI API 서비스
"""
import requests
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class GeminiService:
    """Gemini API 서비스 클래스"""

    def __init__(self):
        """생성자 - 설정은 메서드 호출 시 전달받음"""
        pass

    async def test_connection(self, api_key: str, base_url: str = "https://generativelanguage.googleapis.com/v1beta", model: str = "gemini-2.5-flash") -> Dict[str, Any]:
        """Gemini API 연결 테스트"""
        if not api_key or not api_key.strip():
            raise ValueError("Gemini API Key is not configured")

        api_key = api_key.strip()
        base_url = base_url.rstrip('/')
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        # Models 엔드포인트로 연결 테스트
        models_url = f"{base_url}/models"

        try:
            response = requests.get(models_url, headers=headers, timeout=10)

            if response.status_code == 200:
                models_data = response.json()
                available_models = [model.get("id", "") for model in models_data.get("data", [])]

                # 간단한 completion 테스트
                completion_test = await self.test_completion(api_key, base_url, model)

                return {
                    "status": "success",
                    "message": "Gemini API connection successful",
                    "api_url": base_url,
                    "models_count": len(available_models),
                    "configured_model": model,
                    "model_available": model in available_models if model else None,
                    "completion_test": completion_test
                }
            elif response.status_code == 401:
                raise ValueError("Invalid Gemini API key")
            else:
                raise requests.exceptions.HTTPError(f"HTTP {response.status_code}: {response.text}")

        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to connect to Gemini API at {base_url}: {str(e)}")

    async def test_completion(self, api_key: str, base_url: str, model: str) -> Dict[str, Any]:
        pass

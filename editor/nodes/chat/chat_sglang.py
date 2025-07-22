import os
from editor.node_composer import Node
from langchain_openai import ChatOpenAI
from config.persistent_config import get_all_persistent_configs

class ChatSGLNode(Node):
    categoryId = "langchain"
    functionId = "chat_models"

    nodeId = "chat/sgl"
    nodeName = "Chat SGLang"
    description = "SGLang 서버의 LLM 모델을 사용하여 텍스트 입력에 대한 대화형 응답을 생성합니다. Qwen, Llama, Mistral 등 다양한 오픈소스 모델을 고성능으로 지원합니다."
    tags = ["sglang", "sgl", "qwen", "llama", "mistral", "chat", "ai", "language_model", "conversation", "text_generation", "opensource", "high_performance"]
    
    inputs = [
        {
            "id": "text", 
            "name": "Text", 
            "type": "STR",
            "multi": False,
            "required": True
        },
    ]
    outputs = [
        {
            "id": "result", 
            "name": "Result", 
            "type": "STR"
        },
    ]
    parameters = [
        {
            "id": "model", 
            "name": "Model", 
            "type": "STR",
            "value": "Qwen/Qwen3-4B",
            "required": True,
            "optional": False,
            "options": [
                {"value": "Qwen/Qwen3-4B", "label": "Qwen3-4B"},
                {"value": "Qwen/Qwen3-7B", "label": "Qwen3-7B"},
                {"value": "Qwen/Qwen3-14B", "label": "Qwen3-14B"},
                {"value": "Qwen/Qwen3-32B", "label": "Qwen3-32B"},
                {"value": "meta-llama/Llama-2-7b-chat-hf", "label": "Llama-2-7b-chat"},
                {"value": "meta-llama/Llama-2-13b-chat-hf", "label": "Llama-2-13b-chat"},
                {"value": "meta-llama/Meta-Llama-3.1-8B-Instruct", "label": "Llama-3.1-8B-Instruct"},
                {"value": "meta-llama/Meta-Llama-3.1-70B-Instruct", "label": "Llama-3.1-70B-Instruct"},
                {"value": "mistralai/Mistral-7B-Instruct-v0.3", "label": "Mistral-7B-Instruct-v0.3"},
                {"value": "mistralai/Mixtral-8x7B-Instruct-v0.1", "label": "Mixtral-8x7B-Instruct"},
                {"value": "google/gemma-2-9b-it", "label": "Gemma-2-9B-IT"},
                {"value": "microsoft/DialoGPT-medium", "label": "DialoGPT-medium"},
                {"value": "custom", "label": "Custom Model (use configured model)"}
            ]
        },
        {
            "id": "temperature",
            "name": "Temperature",
            "type": "FLOAT",
            "value": 0.7,
            "required": False,
            "optional": True,  # Advanced 모드에서만 표시
            "min": 0.0,
            "max": 2.0,
            "step": 0.01
        },
        {
            "id": "max_tokens",
            "name": "Max Tokens",
            "type": "INTEGER",
            "value": 512,
            "required": False,
            "optional": True,  # Advanced 모드에서만 표시
            "min": 1,
            "max": 8192,
            "step": 1
        },
        {
            "id": "top_p",
            "name": "Top P",
            "type": "FLOAT",
            "value": 0.9,
            "required": False,
            "optional": True,  # Advanced 모드에서만 표시
            "min": 0.0,
            "max": 1.0,
            "step": 0.01
        },
        {
            "id": "frequency_penalty",
            "name": "Frequency Penalty",
            "type": "FLOAT",
            "value": 0.0,
            "required": False,
            "optional": True,  # Advanced 모드에서만 표시
            "min": -2.0,
            "max": 2.0,
            "step": 0.01
        },
        {
            "id": "presence_penalty",
            "name": "Presence Penalty",
            "type": "FLOAT",
            "value": 0.0,
            "required": False,
            "optional": True,  # Advanced 모드에서만 표시
            "min": -2.0,
            "max": 2.0,
            "step": 0.01
        },
        {
            "id": "seed",
            "name": "Random Seed",
            "type": "INTEGER",
            "value": -1,
            "required": False,
            "optional": True,  # Advanced 모드에서만 표시
            "min": -1,
            "max": 2147483647,
            "step": 1
        }
    ]

    def _get_sgl_config(self):
        """SGLang 설정 정보를 가져옵니다."""
        configs = get_all_persistent_configs()
        
        config_dict = {}
        for config in configs:
            if config.env_name == "SGL_API_BASE_URL":
                config_dict['base_url'] = config.value
            elif config.env_name == "SGL_API_KEY":
                config_dict['api_key'] = config.value
            elif config.env_name == "SGL_MODEL_NAME":
                config_dict['default_model'] = config.value
            elif config.env_name == "SGL_TEMPERATURE_DEFAULT":
                config_dict['default_temperature'] = config.value
            elif config.env_name == "SGL_MAX_TOKENS_DEFAULT":
                config_dict['default_max_tokens'] = config.value
            elif config.env_name == "SGL_TOP_P":
                config_dict['default_top_p'] = config.value
            elif config.env_name == "SGL_FREQUENCY_PENALTY":
                config_dict['default_frequency_penalty'] = config.value
            elif config.env_name == "SGL_PRESENCE_PENALTY":
                config_dict['default_presence_penalty'] = config.value
            elif config.env_name == "SGL_SEED":
                config_dict['default_seed'] = config.value
            elif config.env_name == "SGL_STOP_SEQUENCES":
                config_dict['stop_sequences'] = config.value
            elif config.env_name == "SGL_REQUEST_TIMEOUT":
                config_dict['timeout'] = config.value
        
        return config_dict

    def _parse_stop_sequences(self, stop_sequences_str):
        """Stop sequences 문자열을 파싱합니다."""
        try:
            if stop_sequences_str:
                import json
                return json.loads(stop_sequences_str)
        except (json.JSONDecodeError, TypeError):
            # 기본 stop sequences 반환
            pass
        return ["</s>", "Human:", "Assistant:", "<|endoftext|>"]

    def execute(self, text: str, model: str, temperature: float = 0.7, max_tokens: int = 512, 
                top_p: float = 0.9, frequency_penalty: float = 0.0, presence_penalty: float = 0.0, 
                seed: int = -1) -> str:
        
        # SGLang 설정 가져오기
        sgl_config = self._get_sgl_config()
        
        base_url = sgl_config.get('base_url', 'http://localhost:12721/v1')
        api_key = sgl_config.get('api_key', 'dummy-key')  # SGLang은 API 키가 선택사항
        
        # 모델 선택 로직
        if model == "custom":
            model = sgl_config.get('default_model', 'Qwen/Qwen3-4B')
        
        # Stop sequences 파싱
        stop_sequences = self._parse_stop_sequences(sgl_config.get('stop_sequences'))
        
        # SGLang 서버가 OpenAI API와 호환되므로 ChatOpenAI 사용
        try:
            # model_kwargs 구성
            model_kwargs = {
                "top_p": top_p,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
                "stop": stop_sequences,
            }
            
            # seed가 -1이 아닌 경우에만 추가 (SGLang에서 -1은 랜덤 시드)
            if seed != -1:
                model_kwargs["seed"] = seed
            
            # timeout 설정이 있는 경우 추가
            timeout = sgl_config.get('timeout')
            if timeout:
                model_kwargs["request_timeout"] = int(timeout)
            
            llm = ChatOpenAI(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                base_url=base_url,
                api_key=api_key,
                model_kwargs=model_kwargs
            )
            
            result = llm.invoke(text)
            return result.content if hasattr(result, 'content') else str(result)
            
        except Exception as e:
            # 에러 발생 시 상세한 정보 제공
            error_msg = f"SGLang 모델 실행 중 오류가 발생했습니다: {str(e)}\n"
            error_msg += f"사용된 설정: base_url={base_url}, model={model}\n"
            
            # 설정 검증 도움말 추가
            if "connection" in str(e).lower() or "timeout" in str(e).lower():
                error_msg += "확인사항:\n"
                error_msg += "1. SGLang 서버가 실행 중인지 확인하세요\n"
                error_msg += "2. API Base URL이 올바른지 확인하세요\n"
                error_msg += "3. 방화벽 설정을 확인하세요\n"
                error_msg += f"4. 모델 '{model}'이 SGLang 서버에 로드되어 있는지 확인하세요"
            
            raise RuntimeError(error_msg)

    def get_model_suggestions(self):
        """SGLang에서 사용 가능한 모델 목록을 동적으로 가져옵니다 (선택사항)."""
        try:
            sgl_config = self._get_sgl_config()
            base_url = sgl_config.get('base_url', 'http://localhost:12721/v1')
            api_key = sgl_config.get('api_key')
            
            import requests
            headers = {"Content-Type": "application/json"}
            if api_key and api_key.strip() and api_key != 'dummy-key':
                headers["Authorization"] = f"Bearer {api_key.strip()}"
            
            models_url = f"{base_url}/models"
            response = requests.get(models_url, headers=headers, timeout=5)
            
            if response.status_code == 200:
                models_data = response.json()
                available_models = [model.get("id", "") for model in models_data.get("data", [])]
                return available_models
            
        except Exception:
            # 에러가 발생하면 기본 모델 목록 사용
            pass
        
        return None  # 기본 파라미터의 options 사용

    def validate_model(self, model: str):
        """모델이 유효한지 검증합니다."""
        if model == "custom":
            return True
        
        # 동적으로 사용 가능한 모델 목록 가져오기
        available_models = self.get_model_suggestions()
        if available_models:
            return model in available_models
        
        # 동적 검증이 실패하면 기본 옵션 목록에서 확인
        default_models = [opt["value"] for opt in self.parameters[0]["options"]]
        return model in default_models
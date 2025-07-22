import os
from editor.node_composer import Node
from langchain_openai import ChatOpenAI
from config.persistent_config import get_all_persistent_configs

class ChatVLLMNode(Node):
    categoryId = "langchain"
    functionId = "chat_models"

    nodeId = "chat/vllm"
    nodeName = "Chat vLLM"
    description = "vLLM 서버의 LLM 모델을 사용하여 텍스트 입력에 대한 대화형 응답을 생성합니다. Llama, Vicuna, CodeLlama 등 다양한 오픈소스 모델을 지원합니다."
    tags = ["vllm", "llama", "vicuna", "chat", "ai", "language_model", "conversation", "text_generation", "opensource"]
    
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
            "value": "meta-llama/Llama-2-7b-chat-hf",
            "required": True,
            "optional": False,
            "options": [
                {"value": "meta-llama/Llama-2-7b-chat-hf", "label": "Llama-2-7b-chat"},
                {"value": "meta-llama/Llama-2-13b-chat-hf", "label": "Llama-2-13b-chat"},
                {"value": "meta-llama/Llama-2-70b-chat-hf", "label": "Llama-2-70b-chat"},
                {"value": "lmsys/vicuna-7b-v1.5", "label": "Vicuna-7b-v1.5"},
                {"value": "lmsys/vicuna-13b-v1.5", "label": "Vicuna-13b-v1.5"},
                {"value": "codellama/CodeLlama-7b-Instruct-hf", "label": "CodeLlama-7b-Instruct"},
                {"value": "mistralai/Mistral-7B-Instruct-v0.1", "label": "Mistral-7B-Instruct"},
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
            "step": 0.1
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
            "id": "repetition_penalty",
            "name": "Repetition Penalty",
            "type": "FLOAT",
            "value": 1.0,
            "required": False,
            "optional": True,  # Advanced 모드에서만 표시
            "min": 0.1,
            "max": 2.0,
            "step": 0.01
        }
    ]

    def _get_vllm_config(self):
        """vLLM 설정 정보를 가져옵니다."""
        configs = get_all_persistent_configs()
        
        config_dict = {}
        for config in configs:
            if config.env_name == "VLLM_API_BASE_URL":
                config_dict['base_url'] = config.value
            elif config.env_name == "VLLM_API_KEY":
                config_dict['api_key'] = config.value
            elif config.env_name == "VLLM_MODEL_NAME":
                config_dict['default_model'] = config.value
            elif config.env_name == "VLLM_TEMPERATURE_DEFAULT":
                config_dict['default_temperature'] = config.value
            elif config.env_name == "VLLM_MAX_TOKENS_DEFAULT":
                config_dict['default_max_tokens'] = config.value
        
        return config_dict

    def execute(self, text: str, model: str, temperature: float = 0.7, max_tokens: int = 512, 
                top_p: float = 0.9, repetition_penalty: float = 1.0) -> str:
        
        # vLLM 설정 가져오기
        vllm_config = self._get_vllm_config()
        
        base_url = vllm_config.get('base_url', 'http://localhost:8000/v1')
        api_key = vllm_config.get('api_key', 'dummy-key')  # vLLM은 API 키가 필요 없지만 langchain에서 요구할 수 있음
        
        # 모델 선택 로직
        if model == "custom":
            model = vllm_config.get('default_model', 'meta-llama/Llama-2-7b-chat-hf')
        
        # vLLM 서버가 OpenAI API와 호환되므로 ChatOpenAI 사용
        try:
            llm = ChatOpenAI(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                base_url=base_url,
                api_key=api_key,
                model_kwargs={
                    "top_p": top_p,
                    "repetition_penalty": repetition_penalty,
                    "stop": ["</s>", "<|endoftext|>"],  # 일반적인 stop sequences
                }
            )
            
            result = llm.invoke(text)
            return result.content if hasattr(result, 'content') else str(result)
            
        except Exception as e:
            # 에러 발생 시 상세한 정보 제공
            error_msg = f"vLLM 모델 실행 중 오류가 발생했습니다: {str(e)}\n"
            error_msg += f"사용된 설정: base_url={base_url}, model={model}"
            raise RuntimeError(error_msg)
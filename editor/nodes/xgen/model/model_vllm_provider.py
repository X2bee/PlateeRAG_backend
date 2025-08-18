from fastapi import Request
from typing import Dict, Any
from editor.node_composer import Node
from editor.utils.helper.service_helper import AppServiceManager
from langchain_openai import ChatOpenAI

class ModelVLLMProvider(Node):
    categoryId = "xgen"
    functionId = "chat_models"
    nodeId = "chat_models/vllm"
    nodeName = "Model VLLM"
    description = "VLLM Model을 제공하는 노드입니다."
    tags = ["model", "vllm"]

    inputs = [
    ]
    outputs = [
        {"id": "model", "name": "Model", "type": "MODEL"},
    ]
    parameters = [
        {"id": "model", "name": "Model", "type": "STR", "value": "", "is_api": True, "api_name": "api_vllm_model_name", "required": True},
        {"id": "temperature", "name": "Temperature", "type": "FLOAT", "value": 0.7, "required": False, "optional": True, "min": 0.0, "max": 2.0, "step": 0.1},
        {"id": "max_tokens", "name": "Max Tokens", "type": "INT", "value": 8192, "required": False, "optional": True, "min": 1, "max": 65536, "step": 1},
        {"id": "base_url", "name": "Base URL", "type": "STR", "value": "", "is_api": True, "api_name": "api_vllm_api_base_url", "required": True},
        {"id": "streaming", "name": "Streaming", "type": "BOOL", "value": False, "required": False, "optional": True}
    ]

    def api_vllm_model_name(self, request: Request) -> Dict[str, Any]:
        config_composer = request.app.state.config_composer
        return config_composer.get_config_by_name("VLLM_MODEL_NAME").value

    def api_vllm_api_base_url(self, request: Request) -> Dict[str, Any]:
        config_composer = request.app.state.config_composer
        return config_composer.get_config_by_name("VLLM_API_BASE_URL").value

    def execute(self, model: str = "gpt-5", temperature: float = 0.7, max_tokens: int = 8192, base_url: str = "https://api.openai.com/v1", streaming: bool = False) -> str:
        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=base_url,
            streaming=streaming
        )

        return llm

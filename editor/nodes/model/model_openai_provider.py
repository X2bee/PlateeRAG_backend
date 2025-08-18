from editor.node_composer import Node
from editor.utils.helper.service_helper import AppServiceManager
from langchain_openai import ChatOpenAI

class ModelOpenAIProvider(Node):
    categoryId = "xgen"
    functionId = "chat_models"
    nodeId = "chat_models/openai"
    nodeName = "Model OpenAI"
    description = "OpenAI Model을 제공하는 노드입니다."
    tags = ["model", "openai"]

    inputs = [
    ]
    outputs = [
        {"id": "model", "name": "Model", "type": "MODEL"},
    ]
    parameters = [
        {
            "id": "model", "name": "Model", "type": "STR", "value": "gpt-5", "required": True, "optional": False,
            "options": [
                {"value": "gpt-oss-20b", "label": "GPT-OSS-20B"},
                {"value": "gpt-oss-120b", "label": "GPT-OSS-120B"},
                {"value": "gpt-3.5-turbo", "label": "GPT-3.5 Turbo"},
                {"value": "gpt-4", "label": "GPT-4"},
                {"value": "gpt-4o", "label": "GPT-4o"},
                {"value": "gpt-5", "label": "GPT-5"},
                {"value": "gpt-5-mini", "label": "GPT-5 Mini"},
            ]
        },
        {"id": "temperature", "name": "Temperature", "type": "FLOAT", "value": 0.7, "required": False, "optional": True, "min": 0.0, "max": 2.0, "step": 0.1},
        {"id": "max_tokens", "name": "Max Tokens", "type": "INT", "value": 8192, "required": False, "optional": True, "min": 1, "max": 65536, "step": 1},
        {"id": "base_url", "name": "Base URL", "type": "STR", "value": "https://api.openai.com/v1", "required": False, "optional": True},
        {"id": "streaming", "name": "Streaming", "type": "BOOL", "value": False, "required": False, "optional": True}
    ]

    def execute(self, model: str = "gpt-5", temperature: float = 0.7, max_tokens: int = 8192, base_url: str = "https://api.openai.com/v1", streaming: bool = False) -> str:
        config_composer = AppServiceManager.get_config_composer()
        if not config_composer:
            return "Config Composer가 설정되지 않았습니다."

        api_key = config_composer.get_config_by_name("OPENAI_API_KEY").value

        llm = ChatOpenAI(
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=base_url,
            streaming=streaming
        )

        return llm

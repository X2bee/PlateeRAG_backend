import os
from src.node_composer import Node
from langchain_openai import ChatOpenAI

class ChatOpenAINode(Node):
    categoryId = "langchain"
    functionId = "chat_models"

    nodeId = "chat/openai"
    nodeName = "Chat OpenAI"
    description = "OpenAI의 GPT 모델을 사용하여 텍스트 입력에 대한 대화형 응답을 생성합니다. GPT-3.5, GPT-4, GPT-4o 등의 모델을 선택할 수 있습니다."
    tags = ["openai", "gpt", "chat", "ai", "language_model", "conversation", "text_generation"]
    
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
            "value": "gpt-3.5-turbo",
            "required": True,
            "optional": False,
            "options": [
                {"value": "gpt-3.5-turbo", "label": "GPT-3.5 Turbo"},
                {"value": "gpt-4", "label": "GPT-4"},
                {"value": "gpt-4o", "label": "GPT-4o"}
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
            "value": 1000,
            "required": False,
            "optional": True,  # Advanced 모드에서만 표시
            "min": 1,
            "max": 4000,
            "step": 1
        }
    ]


    def execute(self, text: str, model: str, temperature: float = 0.7, max_tokens: int = 1000) -> str:
        llm = ChatOpenAI(model=model, temperature=temperature, max_tokens=max_tokens)
        result = llm.invoke(text)
        return result.content if hasattr(result, 'content') else str(result)
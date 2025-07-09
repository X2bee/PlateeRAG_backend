import os
from src.node_composer import Node
from langchain_openai import ChatOpenAI

class ChatOpenAINode(Node):
    categoryId = "langchain"
    functionId = "chat_models"

    nodeId = "chat/openai"
    nodeName = "Chat OpenAI"
    
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
            "options": [
                {"value": "gpt-3.5-turbo", "label": "GPT-3.5 Turbo"},
                {"value": "gpt-4", "label": "GPT-4"},
                {"value": "gpt-4o", "label": "GPT-4o"}
            ]
        }
    ]

    def execute(self, text: str, model: str) -> str:
        llm = ChatOpenAI(model=model, temperature=0.7, max_tokens=1000)
        result = llm.invoke(text)
        return result.content if hasattr(result, 'content') else str(result)
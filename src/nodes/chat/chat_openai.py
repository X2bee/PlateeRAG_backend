from src.node_composer import Node

class ChatOpenAINode(Node):
    categoryId = "langchain"
    functionId = "chat_models"

    nodeId = "chat/openai"
    nodeName = "Chat OpenAI"
    
    inputs = [
        {
            "id": "a", 
            "name": "A", 
            "type": "INT",
            "multi": False,
            "required": True
        },
        {
            "id": "b", 
            "name": "B", 
            "type": "INT",
            "multi": False,
            "required": False
        },
    ]
    outputs = [
        {
            "id": "result", 
            "name": "Result", 
            "type": "INT"
        },
    ]
    parameters = []

    def execute(self, a: int, b: int) -> int:
        return a + b
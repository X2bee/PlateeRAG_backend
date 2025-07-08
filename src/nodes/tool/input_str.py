from src.node_composer import (
    Node
)

class InputStringNode(Node):
    categoryId = "utilities"
    functionId = "tools"
    nodeId = "math/input_str"
    nodeName = "Input String"
    
    inputs = []
    outputs = [
        {
            "id": "result", 
            "name": "Result", 
            "type": "STR"
        },
    ]
    parameters = [
        {
            "id": "input_str", 
            "name": "String", 
            "type": "STR",
            "value": "",
        },
    ]

    def execute(self, input_str: int) -> int:
        return input_str
from src.node_composer import (
    Node
)

class InputIntegersNode(Node):
    categoryId = "utilities"
    functionId = "tools"
    nodeId = "math/input_int"
    nodeName = "Input Integer"
    
    inputs = []
    outputs = [
        {
            "id": "result", 
            "name": "Result", 
            "type": "INT"
        },
    ]
    parameters = [
        {
            "id": "input_int", 
            "name": "Integer", 
            "type": "INT",
            "value": 0,
            "step": 1,
            "min": -2147483648,
            "max": 2147483647
        },
    ]

    def execute(self, input_int: int) -> int:
        return input_int
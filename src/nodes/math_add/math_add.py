# backend/nodes/math/add.py
from src.node_composer import (
    Node
)

class AddIntegersNode(Node):
    categoryId = "math"
    categoryName = "Math"
    
    functionId = "arithmetic"
    functionName = "Arithmetic"

    nodeId = "math/add_integers"
    nodeName = "Add Integers"
    
    inputs = [
        {
            "id": "a", 
            "name": "A", 
            "type": "INT",
            "multi": False
        },
        {
            "id": "b", 
            "name": "B", 
            "type": "INT",
            "multi": False
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
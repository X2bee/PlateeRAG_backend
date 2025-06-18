# backend/nodes/math/add.py

from src.node_composer import BaseNode

class AddIntegersNode(BaseNode):
    categoryId = "math"
    categoryName = "Math"
    
    functionId = "arithmetic"
    functionName = "Arithmetic"

    nodeId = "math/add_integers"
    nodeName = "Add Integers"
    
    inputs = [
        {"id": "a", "name": "A", "type": "INT"},
        {"id": "b", "name": "B", "type": "INT"},
    ]
    outputs = [
        {"id": "result", "name": "Result", "type": "INT"},
    ]
    parameters = []

    def execute(self, a: int, b: int) -> int:
        return a + b
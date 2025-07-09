from src.node_composer import Node

class AddIntegersNode(Node):
    categoryId = "math"
    functionId = "arithmetic"

    nodeId = "math/add_integers"
    nodeName = "Add Integers"
    description = "두 개의 정수를 입력받아 더한 결과를 반환합니다. 기본적인 수학 연산 노드입니다."
    tags = ["math", "arithmetic", "addition", "integer", "calculation", "basic_operation"]
    
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
            "required": True
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
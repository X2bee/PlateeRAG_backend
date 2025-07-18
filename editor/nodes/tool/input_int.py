from editor.node_composer import Node

class InputIntegersNode(Node):
    categoryId = "utilities"
    functionId = "startnode"
    nodeId = "math/input_int"
    nodeName = "Input Integer"
    description = "사용자가 설정한 정수 값을 출력하는 입력 노드입니다. 워크플로우에서 숫자 데이터의 시작점으로 사용됩니다."
    tags = ["input", "integer", "number", "parameter", "source", "start_node", "user_input"]
    
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
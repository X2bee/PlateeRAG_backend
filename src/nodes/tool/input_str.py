from src.node_composer import Node

class InputStringNode(Node):
    categoryId = "utilities"
    functionId = "startnode"
    nodeId = "math/input_str"
    nodeName = "Input String"
    description = "사용자가 설정한 문자열 값을 출력하는 입력 노드입니다. 워크플로우에서 텍스트 데이터의 시작점으로 사용됩니다."
    tags = ["input", "string", "text", "parameter", "source", "start_node", "user_input"]
    
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
from editor.node_composer import Node

class PrintAnyNode(Node):
    categoryId = "utilities"
    functionId = "endnode"
    nodeId = "tools/print_any"
    nodeName = "Print Any"
    description = "임의의 타입의 데이터를 입력받아 그대로 반환하는 출력 노드입니다. 워크플로우의 최종 결과를 확인하는데 사용됩니다."
    tags = ["output", "print", "display", "debug", "end_node", "utility", "any_type"]

    inputs = [
        {
            "id": "input_print",
            "name": "Print",
            "type": "ANY",
            "multi": False,
            "required": True
        },
    ]

    def execute(self, input_print: any) -> int:
        return input_print

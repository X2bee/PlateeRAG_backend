from src.node_composer import Node

class PrintAnyNode(Node):
    categoryId = "utilities"
    functionId = "endnode"

    nodeId = "tools/print_any"
    nodeName = "Print Any"
    
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
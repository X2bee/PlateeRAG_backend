from src.node_composer import Node

class TestInvalidNode(Node):
    """
    테스트용 노드 - required=True이면서 optional=True인 잘못된 파라미터를 가짐
    이 노드는 등록에 실패해야 함
    """
    categoryId = "utilities"
    functionId = "tools"
    nodeId = "test/invalid"
    nodeName = "Test Invalid Node"
    description = "테스트용 잘못된 노드 - 등록되지 않아야 함"
    tags = ["test", "invalid", "validation"]
    
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
            "id": "invalid_param",
            "name": "Invalid Parameter",
            "type": "STR",
            "value": "test",
            "required": True,   # 이것과
            "optional": True    # 이것이 동시에 True이면 안됨
        }
    ]

    def execute(self, invalid_param: str) -> str:
        return invalid_param


class TestValidNode(Node):
    """
    테스트용 노드 - 올바른 파라미터 설정을 가짐
    이 노드는 정상적으로 등록되어야 함
    """
    categoryId = "utilities"
    functionId = "tools"
    nodeId = "test/valid"
    nodeName = "Test Valid Node"
    description = "테스트용 올바른 노드 - 정상 등록되어야 함"
    tags = ["test", "valid", "validation"]
    
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
            "id": "basic_param",
            "name": "Basic Parameter",
            "type": "STR",
            "value": "default",
            "required": True,
            "optional": False  # 기본 파라미터
        },
        {
            "id": "advanced_param",
            "name": "Advanced Parameter",
            "type": "STR",
            "value": "advanced_default",
            "required": False,
            "optional": True   # 고급 파라미터
        }
    ]

    def execute(self, basic_param: str, advanced_param: str = "advanced_default") -> str:
        return f"{basic_param} - {advanced_param}"

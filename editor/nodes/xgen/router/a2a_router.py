from editor.node_composer import Node
import logging

logger = logging.getLogger('RouterNode')

class A2ARouterNode(Node):
    categoryId = "xgen"
    functionId = "router"
    nodeId = "router/A2A_Router"
    nodeName = "A2A Router"
    description = "Agent끼리의 연결을 위한 Router입니다. Streaming 등 문자 처리를 용이하게 합니다."
    tags = ["router", "conditional", "branching", "flow-control"]

    inputs = [
        {"id": "agent_output", "name": "Input Data", "type": "ANY", "multi": False, "required": True},
    ]
    outputs = [
        {"id": "default", "name": "Default", "type": "ANY", "multi": False, "required": False},
    ]

    def execute(self, *args, **kwargs):
        """
        라우팅 로직을 실행합니다.

        Args:
            agent_output: 입력 데이터 (Dict 형태여야 함)
            routing_criteria: 라우팅 기준이 되는 키 이름

        Returns:
            입력 데이터를 그대로 반환 (라우팅은 WorkflowExecutor에서 처리)
        """
        agent_output = kwargs.get('agent_output')

        return agent_output

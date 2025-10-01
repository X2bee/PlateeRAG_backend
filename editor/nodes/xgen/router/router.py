from editor.node_composer import Node
import logging

logger = logging.getLogger('RouterNode')

class RouterNode(Node):
    categoryId = "xgen"
    functionId = "router"
    nodeId = "router/Router"
    nodeName = "Router"
    description = "주어진 Dict 입력에서 routing_criteria에 해당하는 키의 값에 따라 다양한 경로로 라우팅하는 노드입니다. 해당 값과 일치하는 output id로만 데이터를 전달합니다."
    tags = ["router", "conditional", "branching", "flow-control"]

    inputs = [
        {"id": "agent_output", "name": "Input Data", "type": "ANY", "multi": False, "required": True},
    ]
    outputs = [
        # 동적으로 프론트엔드에서 추가/제거 가능한 출력 포트들
        # 예: {"id": "True", "name": "True Route", "type": "ANY", "multi": False, "required": False}
        # 예: {"id": "False", "name": "False Route", "type": "ANY", "multi": False, "required": False}
    ]

    parameters = [
        {"id": "routing_criteria", "name": "Routing Criteria", "type": "STR", "value": "", "required": True, "description": "라우팅 기준이 되는 Dict의 키 이름을 설정합니다. 예: 'is_human'"},
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
        routing_criteria = kwargs.get('routing_criteria', '')

        logger.info("RouterNode 실행 - routing_criteria: %s", routing_criteria)
        logger.info("agent_output type: %s, value: %s", type(agent_output), agent_output)

        # 그냥 원본 데이터를 반환
        # 라우팅 로직은 WorkflowExecutor에서 routing_criteria를 참조하여 처리
        logger.info("RouterNode 완료 - 원본 데이터 반환")

        return agent_output

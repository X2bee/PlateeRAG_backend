import logging
import os
from editor.node_composer import Node
from langchain_community.agent_toolkits import SlackToolkit

logger = logging.getLogger(__name__)

class SlackMCP(Node):
    categoryId = "xgen"
    functionId = "mcp"
    nodeId = "mcp/slack_mcp"
    nodeName = "Slack MCP"
    description = "MCP Server for the Slack API"
    tags = ["mcp", "slack", "messaging", "collaboration", "api"]

    inputs = [
    ]
    outputs = [
        {"id": "tools", "name": "Tools", "type": "TOOL"},
    ]

    parameters = [
        {"id": "slack_user_token", "name": "Slack User Token", "type": "STR", "value": "", "required": True, "description": "Slack User API 토큰입니다. (xoxp-로 시작)"},
    ]

    def execute(self, *args, **kwargs):
        try:
            # 파라미터 추출
            slack_user_token = kwargs.get("slack_user_token", "")

            if not slack_user_token:
                raise ValueError("Slack User Token이 필요합니다.")

            # Slack User Token을 환경 변수로 설정
            os.environ["SLACK_USER_TOKEN"] = slack_user_token

            # SlackToolkit 초기화
            slack_toolkit = SlackToolkit()

            # 모든 Slack 도구들을 가져옴
            tools = slack_toolkit.get_tools()

            logger.info("Slack MCP 도구가 성공적으로 생성되었습니다. (%d개의 도구)", len(tools))

            # 도구가 하나만 있는 경우 해당 도구를 반환, 여러 개인 경우 첫 번째 도구를 반환
            return tools[0] if tools else None

        except Exception as e:
            logger.error("Slack MCP 도구 생성 중 오류 발생: %s", str(e))
            raise e

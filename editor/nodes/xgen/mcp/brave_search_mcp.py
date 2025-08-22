import logging
import os
from editor.node_composer import Node
from langchain_community.tools import BraveSearch

logger = logging.getLogger(__name__)

class BraveSearchMCP(Node):
    categoryId = "xgen"
    functionId = "mcp"
    nodeId = "mcp/brave_search_mcp"
    nodeName = "Brave Search MCP"
    description = "MCP server that integrates the Brave Search API - a real-time API to access web search capabilities"
    tags = ["mcp", "search", "brave", "web", "real-time"]

    inputs = [
    ]
    outputs = [
        {"id": "tools", "name": "Tools", "type": "TOOL"},
    ]

    parameters = [
        {"id": "brave_api_key", "name": "Brave API Key", "type": "STR", "value": "", "required": True, "description": "Brave Search API를 호출하기 위한 API 키입니다."},
        {"id": "count", "name": "Count", "type": "INT", "value": 10, "required": False, "optional": True, "description": "Brave Search API에서 반환할 최대 결과 수입니다."},
        {"id": "country", "name": "Country", "type": "STR", "value": "kr", "required": False, "optional": True, "description": "검색 결과의 국가 코드입니다. (예: kr, us, jp)"},
        {"id": "freshness", "name": "Freshness", "type": "SELECT", "value": "", "required": False, "optional": True, "options": ["", "pd", "pw", "pm", "py"], "description": "검색 결과의 신선도입니다. (pd: 지난 하루, pw: 지난 주, pm: 지난 달, py: 지난 년)"},
        {"id": "text_decorations", "name": "Text Decorations", "type": "BOOL", "value": True, "required": False, "optional": True, "description": "텍스트 데코레이션을 포함할지 여부입니다."},
    ]

    def execute(self, *args, **kwargs):
        try:
            # 파라미터 추출
            brave_api_key = kwargs.get("brave_api_key", "")
            count = kwargs.get("count", 10)
            country = kwargs.get("country", "kr")
            freshness = kwargs.get("freshness", "")
            text_decorations = kwargs.get("text_decorations", True)

            # Brave API Key를 환경 변수로 설정
            os.environ["BRAVE_SEARCH_API_KEY"] = brave_api_key

            # search_kwargs 구성
            search_kwargs = {
                "count": count,
                "country": country,
                "text_decorations": text_decorations
            }

            # freshness가 설정되어 있으면 추가
            if freshness:
                search_kwargs["freshness"] = freshness

            # BraveSearch 도구 생성
            brave_search_tool = BraveSearch.from_api_key(
                api_key=brave_api_key,
                search_kwargs=search_kwargs
            )

            logger.info("Brave Search MCP 도구가 성공적으로 생성되었습니다. (count: %s, country: %s)",
                       count, country)
            return brave_search_tool

        except Exception as e:
            logger.error("Brave Search MCP 도구 생성 중 오류 발생: %s", str(e))
            raise e

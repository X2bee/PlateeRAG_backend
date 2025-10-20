from duckduckgo_search import DDGS
import logging
import os
from editor.node_composer import Node
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

class MetaSearchMCP(Node):
    categoryId = "xgen"
    functionId = "mcp"
    nodeId = "mcp/meta_search_mcp"
    nodeName = "Meta Search MCP"
    description = "Meta Search를 위한 MCP 노드입니다. 사용자 요청에 대해 적절한 사이트를 조사한 뒤, 이를 바탕으로 크롤링하여 응답합니다."
    tags = ["mcp", "search", "rag", "setup"]

    inputs = [
    ]
    outputs = [
        {"id": "tools", "name": "Tools", "type": "TOOL"},
    ]

    parameters = [
        {"id": "max_results", "name": "Max Results", "type": "INT", "value": 7, "required": False, "optional": True, "description": "Meta Search에서 반환할 최대 결과 수입니다."},
    ]

    def execute(self, max_results: int = 7, *args, **kwargs):
        def create_meta_search_tool():
            description = "해당 도구는 메타 검색 도구입니다. 사용자가 트렌드, 이슈, 뉴스 등 풍부한 정보를 요청하는 경우, 해당 도구를 사용하여 관련된 정보를 수집하도록 합니다."
            @tool("meta_search_mcp", description=description)
            def meta_search_tool(query: str) -> str:
                results = DDGS().text(query, region="kr-kr", max_results=max_results)

            return meta_search_tool

        return create_meta_search_tool()

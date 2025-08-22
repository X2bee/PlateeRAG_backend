import logging
import os
from editor.node_composer import Node
from langchain_community.tools.tavily_search import TavilySearchResults
logger = logging.getLogger(__name__)

class TavilySearchMCP(Node):
    categoryId = "xgen"
    functionId = "mcp"
    nodeId = "mcp/tavily_search_mcp"
    nodeName = "Tavily Search MCP"
    description = "Tavily Search를 위한 MCP 노드"
    tags = ["mcp", "search", "rag", "setup"]

    inputs = [
    ]
    outputs = [
        {"id": "tools", "name": "Tools", "type": "TOOL"},
    ]

    parameters = [
        {"id": "tavily_api_key", "name": "Tavily API Key", "type": "STR", "value": "", "required": True, "description": "Tavily API를 호출하기 위한 API 키입니다."},
    ]

    def execute(self, tavily_api_key: str, max_results: int, *args, **kwargs):
        os.environ["TAVILY_API_KEY"] = tavily_api_key
        tavily_search_tool = TavilySearchResults(description="Tavily Search Tool", max_results=max_results)

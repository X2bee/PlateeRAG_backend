import logging
import os
from typing import Optional
from editor.node_composer import Node
from langchain.agents import tool
import requests
import json
from pydantic import BaseModel, Field
from editor.utils.tools.product_query_tool import _product_search_tool

logger = logging.getLogger(__name__)

class ProductSearchMCP(Node):
    categoryId = "xgen"
    functionId = "mcp"
    nodeId = "mcp/product_search_mcp"
    nodeName = "Product Search MCP"
    description = "주어진 Product에 대한 정보를 검색하는 도구입니다."
    tags = ["mcp", "data-lab", "rag", "setup"]

    inputs = []
    outputs = [
        {"id": "tools", "name": "Tools", "type": "TOOL"},
    ]

    parameters = [
        {"id": "description", "name": "Description", "type": "STR", "value": "해당 도구는 특정 상품에 대한 판매 정보를 검색하는 도구입니다.", "required": True, "expandable": True, "description": "이 도구를 언제 사용하여야 하는지 설명합니다. AI는 해당 설명을 통해, 해당 도구를 언제 호출해야할지 결정할 수 있습니다."},
        {"id": "select_type", "name": "Select Type", "type": "STR", "value": "", "required": False, "optional": True, "description": "검색 유형을 선택합니다. 'popular'는 인기 상품, 'future'는 예정된 판매 방송, 'past'는 지난 판매 방송, 'sales'는 현재 판매 중 상품을 의미합니다. 지정하지 않는 경우 AI가 상황에 맞게 결정합니다."},
        {"id": "limit", "name": "Limit", "type": "INT", "value": 10, "required": False, "optional": True, "description": "검색 결과 개수 제한. 기본값은 10입니다."},
        {"id": "include_image", "name": "Include Image", "type": "BOOL", "value": False, "required": False, "optional": True, "description": "검색 결과에 이미지 정보를 포함할지 여부. 기본값은 False입니다."},
    ]

    def execute(self, description: str, select_type: str = "popular", limit: int = 10, include_image: bool = False):
        if description.strip() == "" or len(description.strip()) < 1:
            description = "해당 도구는 특정 상품에 대한 판매 정보를 검색하는 도구입니다."

        if select_type.strip() == "" or len(select_type.strip()) < 1:
            user_selected_type = ""
        else:
            if select_type not in ["popular", "future", "past", "sales"]:
                logger.info(f"Invalid type '{select_type}' provided. Defaulting to 'popular'.")
                select_type = "popular"
            user_selected_type = select_type

        def create_product_search_tool():
            class ProductSearch(BaseModel):
                query: str = Field(description="사용자가 정보를 요청한 상품을 입력하세요.")
                search_type: str = Field(
                    description="검색 유형을 선택합니다. 반드시 'popular', 'future', 'past', 'sales' 중 하나여야만 합니다. 'popular'는 인기 상품, 'future'는 예정된 판매 방송, 'past'는 지난 판매 방송, 'sales'는 현재 판매 중 상품을 의미합니다.",
                    enum=["popular", "future", "past", "sales"]
                )
            @tool("product_search_tool", description=description, args_schema=ProductSearch)
            def product_search_tool(query, search_type: str) -> str:
                try:
                    if user_selected_type and user_selected_type in ["popular", "future", "past", "sales"] and user_selected_type != search_type:
                        search_type = user_selected_type
                    result = _product_search_tool(query, search_type=search_type, limit=limit, include_image=include_image)
                    return result if result else "상품 정보를 가져오지 못했습니다."

                except Exception as e:
                    logger.error(f"Error in trend_rank_tool: {e}")
                    return "인기 검색어 정보를 가져오는 중 오류가 발생했습니다."

            return product_search_tool
        return create_product_search_tool()

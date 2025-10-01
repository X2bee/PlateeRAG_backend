import logging
import os
from typing import Optional
from editor.node_composer import Node
from langchain.agents import tool
import requests
import json
from pydantic import BaseModel, Field
from editor.utils.tools.datalab_tool import get_naver_shopping_insight

logger = logging.getLogger(__name__)

class NaverDatalabMCP(Node):
    categoryId = "xgen"
    functionId = "mcp"
    nodeId = "mcp/naver_datalab_mcp"
    nodeName = "Naver Datalab MCP"
    description = "Naver Datalab을 위한 MCP 노드"
    tags = ["mcp", "data-lab", "rag", "setup"]

    inputs = []
    outputs = [
        {"id": "tools", "name": "Tools", "type": "TOOL"},
    ]

    parameters = [
        {"id": "description", "name": "Description", "type": "STR", "value": "해당 도구는 최신, 분야별 인기 검색어를 파악하는 도구입니다. 반드시 '패션의류', '패션잡화', '화장품/미용', '디지털/가전', '가구/인테리어', '출산/육아', '식품', '스포츠/레저', '생활/건강', '여가/생활편의', '도서' 중 하나의 카테고리를 선택하여야 합니다.", "required": True, "expandable": True, "description": "이 도구를 언제 사용하여야 하는지 설명합니다. AI는 해당 설명을 통해, 해당 도구를 언제 호출해야할지 결정할 수 있습니다."},
    ]

    def execute(self, description: str):
        if description.strip() == "" or len(description.strip()) < 1:
            description = "해당 도구는 최신, 분야별 인기 검색어를 파악하는 도구입니다. 반드시 '패션의류', '패션잡화', '화장품/미용', '디지털/가전', '가구/인테리어', '출산/육아', '식품', '스포츠/레저', '생활/건강', '여가/생활편의', '도서' 중 하나의 카테고리를 선택하여야 합니다."

        def create_trend_rank_tool():
            class ShopTrendRank(BaseModel):
                category_name: str = Field(
                    description="인기 검색어를 파악할 카테고리를 선택하세요. 반드시 제시된 카테고리 중 하나여야만 합니다.",
                    enum=["패션의류", "패션잡화", "화장품/미용", "디지털/가전", "가구/인테리어", "출산/육아", "식품", "스포츠/레저", "생활/건강", "여가/생활편의", "도서"]
                )
            @tool("trend_rank_tool", description=description, args_schema=ShopTrendRank)
            def trend_rank_tool(category_name: str) -> str:
                """
                Naver Shopping Insight API를 사용하여 특정 카테고리의 인기 검색어를 가져옵니다.

                Args:
                    category_name: 인기 검색어를 파악할 카테고리명

                Returns:
                    str: 포맷팅된 인기 검색어 리스트
                """
                try:
                    result = get_naver_shopping_insight(category_name=category_name, recent_days=1)
                    return result if result else "인기 검색어 정보를 가져오지 못했습니다."

                except Exception as e:
                    logger.error(f"Error in trend_rank_tool: {e}")
                    return "인기 검색어 정보를 가져오는 중 오류가 발생했습니다."

            return trend_rank_tool
        return create_trend_rank_tool()

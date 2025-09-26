import logging
import os
import requests
from typing import Optional
from editor.node_composer import Node
from langchain.agents import tool

logger = logging.getLogger(__name__)

class NaverNewsMCP(Node):
    categoryId = "xgen"
    functionId = "mcp"
    nodeId = "mcp/naver_news_mcp"
    nodeName = "Naver News MCP"
    description = "Naver News를 위한 MCP 노드"
    tags = ["mcp", "search", "rag", "setup"]

    inputs = []
    outputs = [
        {"id": "tools", "name": "Tools", "type": "TOOL"},
    ]

    parameters = [
        {"id": "description", "name": "Description", "type": "STR", "value": "해당 도구는 뉴스 검색 도구입니다. 사용자가 트렌드, 이슈, 뉴스 등 풍부한 정보를 요청하는 경우, 해당 도구를 사용하여 관련된 정보를 수집하도록 합니다.", "required": True, "expandable": True, "description": "이 도구를 언제 사용하여야 하는지 설명합니다. AI는 해당 설명을 통해, 해당 도구를 언제 호출해야할지 결정할 수 있습니다."},
        {"id": "naver_client_id", "name": "Naver Client ID", "type": "STR", "value": "", "required": True, "description": "네이버 애플리케이션 등록 시 발급받은 클라이언트 ID입니다."},
        {"id": "naver_client_secret", "name": "Naver Client Secret", "type": "STR", "value": "", "required": True, "description": "네이버 애플리케이션 등록 시 발급받은 클라이언트 시크릿입니다."},
        {"id": "sort", "name": "Sort", "type": "STR", "value": "sim", "required": False, "optional": True, "description": "검색 결과 정렬 방법 ('sim': 정확도순, 'date': 날짜순)"},
    ]

    def execute(self, description: str, naver_client_id: str, naver_client_secret: str, sort: str = 'sim'):
        if description.strip() == "" or len(description.strip()) < 1:
            description = "해당 도구는 뉴스 검색 도구입니다. 사용자가 트렌드, 이슈, 뉴스 등 풍부한 정보를 요청하는 경우, 해당 도구를 사용하여 관련된 정보를 수집하도록 합니다."

        def create_naver_news_tool():
            @tool("naver_news_search_tool", description=description)
            def naver_news_search_tool(query: str, display: int = 10) -> str:
                get_sort = sort
                try:
                    start = 1
                    # 파라미터 검증
                    if display < 5:
                        display = 5
                    if display > 100:
                        display = 100
                    if start > 1000:
                        start = 1000
                    if get_sort not in ['sim', 'date']:
                        get_sort = 'sim'

                    # API 요청 URL 및 헤더 설정
                    url = "https://openapi.naver.com/v1/search/news.json"
                    headers = {
                        "X-Naver-Client-Id": naver_client_id,
                        "X-Naver-Client-Secret": naver_client_secret
                    }
                    params = {
                        "query": query,
                        "display": display,
                        "start": start,
                        "sort": get_sort
                    }

                    # API 요청
                    response = requests.get(url, headers=headers, params=params)
                    response.raise_for_status()

                    # JSON 파싱
                    data = response.json()
                    items = data.get('items', [])

                    if not items:
                        return f"검색어 '{query}'에 대한 뉴스를 찾을 수 없습니다."

                    # 결과 포맷팅
                    results = []
                    for i, item in enumerate(items, 1):
                        title = item.get('title', '제목 없음')
                        link = item.get('link', '')
                        description = item.get('description', '설명 없음')
                        pubDate = item.get('pubDate', '날짜 없음')

                        # HTML 태그 제거
                        import re
                        title = re.sub('<[^<]+?>', '', title)
                        description = re.sub('<[^<]+?>', '', description)

                        results.append(f"[뉴스 {i}]\n제목: {title}\n설명: {description}\n발행일: {pubDate}\n링크: {link}\n")

                    return "\n".join(results)

                except requests.exceptions.RequestException as e:
                    logger.error(f"네이버 뉴스 API 요청 중 오류: {e}")
                    return f"API 요청 중 오류가 발생했습니다: {str(e)}"
                except ValueError as e:  # JSON 파싱 오류
                    logger.error(f"JSON 파싱 중 오류: {e}")
                    return f"응답 데이터 파싱 중 오류가 발생했습니다: {str(e)}"
                except Exception as e:
                    logger.error(f"네이버 뉴스 검색 중 예상치 못한 오류: {e}")
                    return f"검색 중 오류가 발생했습니다: {str(e)}"

            return naver_news_search_tool

        return create_naver_news_tool()

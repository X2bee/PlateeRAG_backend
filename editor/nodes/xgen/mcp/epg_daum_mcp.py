import logging
import os
import json
from typing import List, Dict, Any
from editor.node_composer import Node
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from service.epg_lite.epg_json import epg_to_json

logger = logging.getLogger(__name__)

class EPGDaumMCP(Node):
    categoryId = "xgen"
    functionId = "mcp"
    nodeId = "mcp/epg_daum_mcp"
    nodeName = "EPG DAUM MCP"
    description = "DAUM EPG 데이터를 가져오는 MCP 노드입니다. 한국 홈쇼핑 채널들의 편성표 정보를 JSON 형태로 제공합니다."
    tags = ["mcp", "epg", "daum", "tv", "schedule"]

    inputs = [
    ]
    outputs = [
        {"id": "tools", "name": "Tools", "type": "TOOL"},
    ]

    parameters = [
        {"id": "use_cache", "name": "Use Cache", "type": "BOOL", "value": True, "required": False, "optional": True, "description": "캐시를 사용할지 여부입니다. True이면 캐시된 데이터를 사용하고, False이면 새로운 데이터를 가져옵니다."},
        # {"id": "target_date", "name": "Target Date", "type": "STRING", "value": "", "required": False, "optional": True, "description": "가져올 EPG 데이터의 날짜입니다. YYYY-MM-DD 형식으로 입력하세요. 비어있으면 오늘 날짜를 사용합니다."},
    ]

    def _get_channel_id_by_name(self, channel_name: str, channels: List[Dict]) -> str:
        """채널명으로 채널 ID를 찾습니다. 대소문자, 띄어쓰기, 특수문자 무시하고 유사 매칭합니다."""
        def normalize_text(text):
            """텍스트 정규화: 소문자 변환, 띄어쓰기/특수문자 제거"""
            import re
            text = text.lower()
            text = re.sub(r'[_\-\s&+]', '', text)  # 띄어쓰기, 언더바, 하이픈, &, + 제거
            return text

        def is_similar(str1, str2, max_diff=1):
            """문자열 유사도 체크 (편집 거리 기반)"""
            if abs(len(str1) - len(str2)) > max_diff:
                return False

            # 간단한 편집거리 계산
            m, n = len(str1), len(str2)
            dp = [[0] * (n + 1) for _ in range(m + 1)]

            for i in range(m + 1):
                dp[i][0] = i
            for j in range(n + 1):
                dp[0][j] = j

            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if str1[i-1] == str2[j-1]:
                        dp[i][j] = dp[i-1][j-1]
                    else:
                        dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

            return dp[m][n] <= max_diff

        normalized_input = normalize_text(channel_name)

        # 1. 정확 매칭 시도
        for channel in channels:
            display_names = channel.get("display-names", [])
            for display_name in display_names:
                normalized_display = normalize_text(display_name)
                if normalized_display == normalized_input:
                    return channel["id"]

        # 2. 유사 매칭 시도 (편집거리 1 이하)
        for channel in channels:
            display_names = channel.get("display-names", [])
            for display_name in display_names:
                normalized_display = normalize_text(display_name)
                if is_similar(normalized_display, normalized_input, max_diff=1):
                    return channel["id"]

        return None

    def _format_time(self, time_str: str) -> str:
        """시간 포맷 변환 (20250925010000 +0900 -> 01:00)"""
        try:
            time_part = time_str.split()[0]  # +0900 부분 제거
            hour = time_part[8:10]
            minute = time_part[10:12]
            return f"{hour}:{minute}"
        except:
            return time_str

    def _format_channel_schedule(self, channel_name: str, epg_data: Dict[str, Any]) -> str:
        """특정 채널의 편성표를 포맷팅합니다."""
        channels = epg_data.get("channels", [])
        programmes = epg_data.get("programmes", [])

        # "all_channels" 옵션인 경우 모든 채널의 편성표 반환
        if channel_name.lower() == "all_channels":
            return self._format_all_channels_schedule(epg_data)

        # 채널 ID 찾기
        channel_id = self._get_channel_id_by_name(channel_name, channels)
        if not channel_id:
            return f"채널 '{channel_name}'을 찾을 수 없습니다."

        # 해당 채널의 프로그램만 필터링
        channel_programmes = [
            prog for prog in programmes
            if prog.get("channel") == channel_id
        ]

        if not channel_programmes:
            return f"채널 '{channel_name}'의 편성 정보가 없습니다."

        # 편성표 포맷팅
        result = f"=== {channel_name} 편성표 ===\n"
        for prog in channel_programmes:
            start_time = self._format_time(prog.get("start", ""))
            title = prog.get("title", {}).get("text", "제목 없음")
            result += f"{start_time}: {title}\n"

        return result

    def _format_all_channels_schedule(self, epg_data: Dict[str, Any]) -> str:
        """모든 채널의 편성표를 포맷팅합니다."""
        channels = epg_data.get("channels", [])
        programmes = epg_data.get("programmes", [])

        if not channels or not programmes:
            return "편성 정보가 없습니다."

        result = "=== 전체 채널 편성표 ===\n\n"

        for channel in channels:
            channel_id = channel["id"]
            channel_name = channel["display-names"][0]

            channel_programmes = [
                prog for prog in programmes
                if prog.get("channel") == channel_id
            ]

            if channel_programmes:
                result += f"=== {channel_name} ===\n"
                for prog in channel_programmes:
                    start_time = self._format_time(prog.get("start", ""))
                    title = prog.get("title", {}).get("text", "제목 없음")
                    result += f"{start_time}: {title}\n"

                result += "\n"

        return result

    def execute(self, use_cache: bool = True, target_date: str = "", **kwargs):
        def create_epg_daum_tool():
            description = "특정 채널의 편성표를 조회하는데 사용하는 도구입니다."
            class EPGChannelSchema(BaseModel):
                channel_name: str = Field(
                    description="조회할 채널명을 선택하세요. 'all_channels'을 선택하면 모든 채널의 편성표를 보여줍니다.",
                    enum=["all_channels", "롯데홈쇼핑", "현대홈쇼핑", "CJ온스타일", "GS SHOP", "NS홈쇼핑", "SK stoa", "kt알파 쇼핑"]
                )

            @tool("epg_daum_mcp", description=description, args_schema=EPGChannelSchema)
            def epg_daum_tool(channel_name: str) -> str:
                """
                DAUM EPG에서 특정 채널의 편성표를 가져옵니다.

                Args:
                    channel_name: 조회할 채널명

                Returns:
                    str: 포맷팅된 채널 편성표
                """
                try:
                    date_param = None
                    epg_data = epg_to_json(
                        provider="DAUM",
                        use_cache=use_cache,
                        target_date=date_param
                    )

                    formatted_schedule = self._format_channel_schedule(channel_name, epg_data)
                    return formatted_schedule

                except Exception as e:
                    return None

            return epg_daum_tool
        return create_epg_daum_tool()

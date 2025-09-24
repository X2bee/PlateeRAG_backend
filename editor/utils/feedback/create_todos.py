import json
import logging
import re
from typing import Dict, List, Optional

from langchain.schema import HumanMessage

logger = logging.getLogger(__name__)


todo_generation_prompt = """
사용자 요청을 분석하여 구체적이고 실행 가능한 TODO 리스트를 생성해주세요.

사용자 요청: {text}

다음 조건을 만족하는 JSON 객체를 생성해주세요:
1. "mode": "direct" 또는 "todo" 중 하나.
   - TODO 없이 한 번에 답변 가능하고 도구가 필요 없다면 "direct"를 선택하세요.
   - 조금이라도 복잡하거나 도구 사용/여러 단계가 필요할 것 같다면 "todo"를 선택하세요.
2. "reason": 선택한 모드에 대한 간단한 설명.
3. "tool_usage": "simple" 또는 "complex".
   - direct 모드에서 도구가 필요하면 반드시 "complex"로 표기하세요.
4. "todos": TODO 객체 배열. direct 모드라도 0개 또는 1개의 TODO를 반환할 수 있습니다.
   - 각 TODO는 구체적이고 실행 가능해야 합니다.
   - 순서대로 나열하세요.
   - 각 TODO마다 "tool_required" 필드에 "simple" 또는 "complex" 중 하나를 기입하세요.

응답 형식 예시:
{{
    "mode": "todo",
    "reason": "여러 단계의 정보 수집이 필요함",
    "tool_usage": "complex",
    "todos": [
        {{
            "id": 1,
            "title": "TODO 제목",
            "description": "구체적인 작업 설명",
            "priority": "high|medium|low",
            "tool_required": "simple|complex"
        }}
    ]
}}
"""


KEYWORDS_REQUIRING_TOOLS = [
    "검색",
    "크롤",
    "수집",
    "데이터",
    "분석",
    "파일",
    "다운로드",
    "업로드",
    "변환",
    "convert",
    "csv",
    "excel",
    "api",
    "로그",
    "코드",
    "테스트",
]

SUMMARY_KEYWORDS = [
    "요약",
    "정리",
    "보고",
    "summary",
    "report",
    "결과 공유",
]


def _safe_json_loads(content: str) -> Optional[Dict]:
    tries = [content]
    try:
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            tries.append(json_match.group())
    except Exception:
        pass

    for candidate in tries:
        try:
            return json.loads(candidate)
        except Exception:
            continue
    return None


def _ensure_todo_structure(text: str, todos: Optional[List[Dict]]) -> List[Dict]:
    if not isinstance(todos, list) or not todos:
        return [
            {
                "id": 1,
                "title": "사용자 요청 처리",
                "description": text,
                "priority": "high",
                "tool_required": "simple",
            }
        ]

    normalized = []
    for idx, todo in enumerate(todos, start=1):
        normalized.append(
            {
                "id": todo.get("id", idx),
                "title": todo.get("title", f"TODO {idx}"),
                "description": todo.get("description", text),
                "priority": todo.get("priority", "medium"),
                "tool_required": "complex"
                if str(todo.get("tool_required", "complex")).lower() == "complex"
                else "simple",
            }
        )
    return normalized


def _keyword_hits(text: str) -> bool:
    lowered = text.lower()
    return any(keyword.lower() in lowered for keyword in KEYWORDS_REQUIRING_TOOLS)


def _contains_summary_task(todos: List[Dict]) -> bool:
    for todo in todos:
        combined = f"{todo.get('title', '')} {todo.get('description', '')}".lower()
        if any(keyword.lower() in combined for keyword in SUMMARY_KEYWORDS):
            return True
    return False


def _append_summary_todo(base_text: str, todos: List[Dict]) -> List[Dict]:
    if _contains_summary_task(todos):
        return todos

    summary_id = (todos[-1]["id"] + 1) if todos else 1
    summary_todo = {
        "id": summary_id,
        "title": "결과 요약 및 보고",
        "description": "앞선 TODO 실행 결과와 핵심 인사이트를 정리하고 보고하세요.",
        "priority": "medium",
        "tool_required": "simple",
    }
    return todos + [summary_todo]


def create_todos(llm, text: str, tools_list: Optional[List] = None) -> Dict:
    """LLM으로 TODO 리스트와 실행 모드 결정"""

    prompt = todo_generation_prompt.format(text=text)
    response = llm.invoke([HumanMessage(content=prompt)])
    content = getattr(response, "content", "") or ""

    parsed = _safe_json_loads(content) or {}

    todos = _ensure_todo_structure(text, parsed.get("todos"))
    mode = parsed.get("mode", "todo")
    reason = parsed.get("reason", "")
    tool_usage = parsed.get("tool_usage", "simple")

    if mode not in ("direct", "todo"):
        mode = "todo"
    if tool_usage not in ("simple", "complex"):
        tool_usage = "simple"

    has_tools = bool(tools_list)
    keyword_hit = _keyword_hits(text)
    multi_step = len(todos) > 1
    any_complex = any(todo.get("tool_required") == "complex" for todo in todos)
    long_request = len(text) > 200

    if mode == "direct":
        if multi_step or long_request:
            mode = "todo"
            if not reason:
                reason = "여러 단계 또는 긴 설명이 필요하여 TODO 모드로 전환"
        else:
            if tool_usage == "complex":
                tool_usage = "complex"
            elif has_tools and (any_complex or keyword_hit):
                tool_usage = "complex"
            else:
                tool_usage = "simple"
    else:  # mode == "todo"
        if not multi_step and not any_complex and not long_request and not keyword_hit:
            mode = "direct"
            tool_usage = "simple"
            if not reason:
                reason = "단일 단계로 충분하고 도구가 필요하지 않음"
        elif has_tools and any_complex:
            tool_usage = "complex"
        else:
            tool_usage = "simple"

    todos = _append_summary_todo(text, todos)

    if mode == "direct" and len(todos) > 1:
        mode = "todo"
        if not reason:
            reason = "최종 결과 요약 단계를 포함하기 위해 TODO 모드로 전환"

    if mode == "todo":
        any_complex = any(todo.get("tool_required") == "complex" for todo in todos)
        tool_usage = "complex" if has_tools and any_complex else "simple"

    return {
        "mode": mode,
        "reason": reason,
        "todos": [] if mode == "direct" else todos,
        "tool_usage": tool_usage,
        "raw_todos": todos,
        "llm_raw": content,
    }

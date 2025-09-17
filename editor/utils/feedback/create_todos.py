import logging
from langchain.schema import HumanMessage
logger = logging.getLogger(__name__)

# 1단계: 사용자 요청을 TODO로 분해
todo_generation_prompt = """
사용자 요청을 분석하여 구체적이고 실행 가능한 TODO 리스트를 생성해주세요.

사용자 요청: {text}

다음 조건을 만족하는 TODO 리스트를 JSON 형식으로 생성해주세요:
1. 각 TODO는 구체적이고 실행 가능해야 합니다
2. 복잡한 작업은 여러 단계로 나누어주세요
3. 순서대로 실행되도록 배열해주세요
4. 마지막은 최종 결과를 생성하는 TODO로 마무리해주세요

응답 형식:
{{
    "todos": [
        {{
            "id": 1,
            "title": "TODO 제목",
            "description": "구체적인 작업 설명",
            "priority": "high|medium|low"
        }},
        ...
    ]
}}
"""

def create_todos(llm, text):
    todo_generation_prompt_filled = todo_generation_prompt.format(text=text)
    # TODO 생성
    
    todo_messages = [HumanMessage(content=todo_generation_prompt_filled)]
    todo_response = llm.invoke(todo_messages)

    # TODO 파싱
    import json
    import re
    try:
        # JSON 파싱 시도
        json_match = re.search(r'\{.*\}', todo_response.content, re.DOTALL)
        if json_match:
            todos_data = json.loads(json_match.group())
            return todos_data.get("todos", [])
        else:
            # fallback: 기본 TODO 생성
            return [{"id": 1, "title": "사용자 요청 처리", "description": text, "priority": "high"}]
    except Exception as e:
        logger.warning(f"TODO 파싱 실패, 기본 TODO 사용: {str(e)}")
        return [{"id": 1, "title": "사용자 요청 처리", "description": text, "priority": "high"}]

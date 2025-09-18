# State definitions for LangGraph
from typing import Annotated, Dict, List, Optional, TypedDict
from langgraph.graph.message import add_messages

class FeedbackState(TypedDict):
    messages: Annotated[List[Dict], add_messages]
    user_input: str
    tool_results: List[Dict]
    feedback_score: int  # 1-10 scale
    iteration_count: int
    final_result: Optional[str]
    requirements_met: bool
    max_iterations: int
    todo_requires_tools: Optional[bool]  # TODO가 도구 사용이 필요한지 여부
    current_todo_id: Optional[int]
    current_todo_title: Optional[str]
    current_todo_index: Optional[int]
    total_todos: Optional[int]

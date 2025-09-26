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
    current_todo_description: Optional[str]
    current_todo_index: Optional[int]
    total_todos: Optional[int]
    execution_mode: Optional[str]
    skip_feedback_eval: Optional[bool]
    remediation_notes: Optional[List[str]]
    seen_results: Optional[List[str]]
    last_result_signature: Optional[str]
    last_result_duplicate: Optional[bool]
    stagnation_count: Optional[int]
    result_frequencies: Optional[Dict[str, int]]
    duplicate_run_length: Optional[int]
    original_user_request: Optional[str]
    previous_results_context: Optional[str]
    todo_directive: Optional[str]

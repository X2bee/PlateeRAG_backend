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
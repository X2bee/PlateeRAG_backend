from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class WorkflowRequest(BaseModel):
    workflow_name: str
    workflow_id: str
    input_data: str = ""
    interaction_id: str = "default"
    selected_collections: Optional[List[str]] = None
    user_id: Optional[int | str] = None
    additional_params: Optional[Dict[str, Dict[str, Any]]] = None

class WorkflowData(BaseModel):
    workflow_name: str
    workflow_id: str
    view: Dict[str, Any]
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    interaction_id: str = "default"

class SaveWorkflowRequest(BaseModel):
    workflow_name: str
    content: WorkflowData

class ConversationRequest(BaseModel):
    """통합 대화/워크플로우 실행 요청 모델"""
    user_input: str
    interaction_id: str
    execution_type: str = "default_mode"  # "default_mode" 또는 "workflow"
    workflow_id: Optional[str] = None
    workflow_name: Optional[str] = None
    selected_collections: Optional[List[str]] = None

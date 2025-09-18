from pydantic import BaseModel, Field
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
    user_id: Optional[int | str] = None

class ConversationRequest(BaseModel):
    """통합 대화/워크플로우 실행 요청 모델"""
    user_input: str
    interaction_id: str
    execution_type: str = "default_mode"  # "default_mode" 또는 "workflow"
    workflow_id: Optional[str] = None
    workflow_name: Optional[str] = None
    selected_collections: Optional[List[str]] = None

# Tester related models
class TesterTestCase(BaseModel):
    """테스터 테스트 케이스 모델"""
    id: int
    input: str
    expected_output: Optional[str] = None

class TesterExecuteRequest(BaseModel):
    """테스터 실행 요청 모델"""
    workflow_name: str
    workflow_id: str
    test_cases: List[TesterTestCase]
    batch_size: int = 5
    interaction_id: str = "batch_test"
    selected_collections: Optional[List[str]] = None
    llm_eval_enabled: Optional[bool] = False
    llm_eval_type: Optional[str] = None
    llm_eval_model: Optional[str] = None

class TesterTestResult(BaseModel):
    """테스터 테스트 결과 모델"""
    id: int
    input: str
    expected_output: Optional[str]
    actual_output: Optional[str]
    status: str  # 'success', 'error'
    execution_time: Optional[int]  # milliseconds
    error: Optional[str]
    llm_eval_score: Optional[float] = None  # LLM 평가 점수 (0.0 ~ 1.0)

class ScoreModelParser(BaseModel):
    llm_eval_score: float = Field(
        description="주어진 데이터를 평가하여 0~1점의 점수로 반환합니다. 소수점 2째 자리까지 표현하십시오 (0.00 ~ 1.00)",
        ge=0.00,
        le=1.00
    )

# Deploy related models
class DeployToggleRequest(BaseModel):
    """배포 상태 토글 요청 모델"""
    enable: bool = Field(description="배포 활성화 여부 (True: 활성화, False: 비활성화)")


class DeployStatusRequest(BaseModel):
    """배포 상태 요청 모델"""
    user_id: Optional[str] = Field(description="사용자 ID")

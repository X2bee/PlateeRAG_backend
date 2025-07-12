"""
성능 지표 관련 데이터 모델
"""
from typing import Dict, Optional
from models.base_model import BaseModel

class WorkflowExecution(BaseModel):
    """워크플로우 실행 기록 모델"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.user_id: Optional[int] = kwargs.get('user_id')
        self.workflow_id: str = kwargs.get('workflow_id', '')
        self.workflow_name: str = kwargs.get('workflow_name', '')
        self.status: str = kwargs.get('status', 'pending')  # pending, running, completed, failed
        self.start_time: Optional[str] = kwargs.get('start_time')
        self.end_time: Optional[str] = kwargs.get('end_time')
        self.execution_time: Optional[float] = kwargs.get('execution_time')  # seconds
        self.node_count: int = kwargs.get('node_count', 0)
        self.nodes_executed: int = kwargs.get('nodes_executed', 0)
        self.error_message: Optional[str] = kwargs.get('error_message')
        self.input_data: Optional[Dict] = kwargs.get('input_data', {})
        self.output_data: Optional[Dict] = kwargs.get('output_data', {})
        self.metadata: Optional[Dict] = kwargs.get('metadata', {})
    
    def get_table_name(self) -> str:
        return "workflow_executions"
    
    def get_schema(self) -> Dict[str, str]:
        return {
            'user_id': 'INTEGER REFERENCES users(id) ON DELETE SET NULL',
            'workflow_id': 'VARCHAR(100) NOT NULL',
            'workflow_name': 'VARCHAR(200) NOT NULL',
            'status': 'VARCHAR(20) NOT NULL',
            'start_time': 'TIMESTAMP',
            'end_time': 'TIMESTAMP',
            'execution_time': 'DECIMAL(10,3)', 
            'node_count': 'INTEGER DEFAULT 0',
            'nodes_executed': 'INTEGER DEFAULT 0',
            'error_message': 'TEXT',
            'input_data': 'TEXT',
            'output_data': 'TEXT',
            'metadata': 'TEXT'
        }

class NodeExecution(BaseModel):
    """노드 실행 기록 모델"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.workflow_execution_id: int = kwargs.get('workflow_execution_id', 0)
        self.node_id: str = kwargs.get('node_id', '')
        self.node_type: str = kwargs.get('node_type', '')
        self.status: str = kwargs.get('status', 'pending')  # pending, running, completed, failed
        self.start_time: Optional[str] = kwargs.get('start_time')
        self.end_time: Optional[str] = kwargs.get('end_time')
        self.execution_time: Optional[float] = kwargs.get('execution_time')  # seconds
        self.input_data: Optional[Dict] = kwargs.get('input_data', {})
        self.output_data: Optional[Dict] = kwargs.get('output_data', {})
        self.error_message: Optional[str] = kwargs.get('error_message')
        self.metadata: Optional[Dict] = kwargs.get('metadata', {})
    
    def get_table_name(self) -> str:
        return "node_executions"
    
    def get_schema(self) -> Dict[str, str]:
        return {
            'workflow_execution_id': 'INTEGER NOT NULL REFERENCES workflow_executions(id) ON DELETE CASCADE',
            'node_id': 'VARCHAR(100) NOT NULL',
            'node_type': 'VARCHAR(100) NOT NULL',
            'status': 'VARCHAR(20) NOT NULL',
            'start_time': 'TIMESTAMP',
            'end_time': 'TIMESTAMP',
            'execution_time': 'DECIMAL(8,3)',  # seconds
            'input_data': 'TEXT',      # JSON string
            'output_data': 'TEXT',     # JSON string
            'error_message': 'TEXT',
            'metadata': 'TEXT'         # JSON string
        }

class SystemMetrics(BaseModel):
    """시스템 성능 지표 모델"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.metric_type: str = kwargs.get('metric_type', '')  # cpu, memory, disk, api_call, etc.
        self.metric_value: float = kwargs.get('metric_value', 0.0)
        self.metric_unit: str = kwargs.get('metric_unit', '')  # %, MB, seconds, count, etc.
        self.timestamp: str = kwargs.get('timestamp', '')
        self.component: Optional[str] = kwargs.get('component')  # openai, workflow, node, etc.
        self.details: Optional[Dict] = kwargs.get('details', {})
    
    def get_table_name(self) -> str:
        return "system_metrics"
    
    def get_schema(self) -> Dict[str, str]:
        return {
            'metric_type': 'VARCHAR(50) NOT NULL',
            'metric_value': 'DECIMAL(15,6) NOT NULL',
            'metric_unit': 'VARCHAR(20)',
            'timestamp': 'TIMESTAMP NOT NULL',
            'component': 'VARCHAR(50)',
            'details': 'TEXT'  # JSON string
        }

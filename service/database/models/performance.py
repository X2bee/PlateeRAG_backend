"""
성능 지표 관련 데이터 모델
"""
from typing import Dict, Optional
from service.database.models.base_model import BaseModel

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

class NodePerformance(BaseModel):
    """개별 노드 성능 측정 데이터 모델"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.user_id: Optional[int] = kwargs.get('user_id')
        self.workflow_name: str = kwargs.get('workflow_name', '')
        self.workflow_id: str = kwargs.get('workflow_id', '')
        self.node_id: str = kwargs.get('node_id', '')
        self.node_name: str = kwargs.get('node_name', '')
        self.timestamp: str = kwargs.get('timestamp', '')
        self.processing_time_ms: float = kwargs.get('processing_time_ms', 0.0)
        self.cpu_usage_percent: float = kwargs.get('cpu_usage_percent', 0.0)
        self.ram_usage_mb: float = kwargs.get('ram_usage_mb', 0.0)
        self.gpu_usage_percent: Optional[float] = kwargs.get('gpu_usage_percent')
        self.gpu_memory_mb: Optional[float] = kwargs.get('gpu_memory_mb')
        self.input_data: Optional[str] = kwargs.get('input_data')  # JSON string
        self.output_data: Optional[str] = kwargs.get('output_data')  # JSON string

    def get_table_name(self) -> str:
        return "node_performance"

    def get_schema(self) -> Dict[str, str]:
        return {
            'user_id': 'INTEGER REFERENCES users(id) ON DELETE SET NULL',
            'workflow_name': 'VARCHAR(200) NOT NULL',
            'workflow_id': 'VARCHAR(100) NOT NULL',
            'node_id': 'VARCHAR(100) NOT NULL',
            'node_name': 'VARCHAR(200) NOT NULL',
            'timestamp': 'TIMESTAMP NOT NULL',
            'processing_time_ms': 'DECIMAL(10,2) NOT NULL',
            'cpu_usage_percent': 'DECIMAL(8,2) DEFAULT 0.0',
            'ram_usage_mb': 'DECIMAL(10,2) DEFAULT 0.0',
            'gpu_usage_percent': 'DECIMAL(8,2)',
            'gpu_memory_mb': 'DECIMAL(10,2)',
            'input_data': 'TEXT',
            'output_data': 'TEXT'
        }

"""
채팅 관련 데이터 모델
"""
from typing import Dict, Optional
from service.database.models.base_model import BaseModel

class WorkflowMeta(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.user_id: Optional[int] = kwargs.get('user_id')
        self.workflow_id: str = kwargs.get('workflow_id', '')
        self.workflow_name: str = kwargs.get('workflow_name', '')
        self.node_count: int = kwargs.get('node_count', 0)
        self.edge_count: int = kwargs.get('edge_count', 0)
        self.has_startnode: bool = kwargs.get('has_startnode', False)
        self.has_endnode: bool = kwargs.get('has_endnode', False)
        self.is_completed: bool = kwargs.get('is_completed', False)
        self.metadata: Optional[Dict] = kwargs.get('metadata', {})

    def get_table_name(self) -> str:
        return "workflow_meta"

    def get_schema(self) -> Dict[str, str]:
        return {
            'user_id': 'INTEGER REFERENCES users(id) ON DELETE SET NULL',
            'workflow_id': 'VARCHAR(100) NOT NULL',
            'workflow_name': 'VARCHAR(200) NOT NULL',
            'node_count': 'INTEGER DEFAULT 0',
            'edge_count': 'INTEGER DEFAULT 0',
            'has_startnode': 'BOOLEAN DEFAULT FALSE',
            'has_endnode': 'BOOLEAN DEFAULT FALSE',
            'is_completed': 'BOOLEAN DEFAULT FALSE',
            'metadata': 'TEXT'  # JSON string
        }

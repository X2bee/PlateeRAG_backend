"""
채팅 관련 데이터 모델
"""
from typing import Dict, Optional
from models.base_model import BaseModel

class ExecutionIO(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.user_id: Optional[int] = kwargs.get('user_id')
        self.workflow_id: str = kwargs.get('workflow_id', '')
        self.workflow_name: str = kwargs.get('workflow_name', '')
        self.input_data: Dict = kwargs.get('input_data', {})
        self.output_data: Dict = kwargs.get('output_data', {})
    
    def get_table_name(self) -> str:
        return "execution_io"
    
    def get_schema(self) -> Dict[str, str]:
        return {
            'user_id': 'INTEGER REFERENCES users(id) ON DELETE SET NULL',
            'workflow_id': 'VARCHAR(100) NOT NULL',
            'workflow_name': 'VARCHAR(200) NOT NULL',
            'input_data': 'TEXT',  # JSON string
            'output_data': 'TEXT'  # JSON string
        }

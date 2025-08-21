"""
채팅 관련 데이터 모델
"""
from typing import Dict, Optional
from service.database.models.base_model import BaseModel

class ExecutionIO(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.user_id: Optional[int] = kwargs.get('user_id')
        self.interaction_id: str = kwargs.get('interaction_id', 'default')
        self.workflow_id: str = kwargs.get('workflow_id', '')
        self.workflow_name: str = kwargs.get('workflow_name', '')
        self.input_data: Dict = kwargs.get('input_data', {})
        self.output_data: Dict = kwargs.get('output_data', {})
        self.expected_output: Optional[str] = kwargs.get('expected_output', None)
        self.llm_eval_score: Optional[float] = kwargs.get('llm_eval_score', None)
        self.test_mode: Optional[bool] = kwargs.get('test_mode', False)

    def get_table_name(self) -> str:
        return "execution_io"

    def get_schema(self) -> Dict[str, str]:
        return {
            'user_id': 'INTEGER REFERENCES users(id) ON DELETE SET NULL',
            'interaction_id': 'VARCHAR(100) NOT NULL DEFAULT \'default\'',
            'workflow_id': 'VARCHAR(100) NOT NULL',
            'workflow_name': 'VARCHAR(200) NOT NULL',
            'input_data': 'TEXT',
            'output_data': 'TEXT',
            'expected_output': 'TEXT',
            'llm_eval_score': 'FLOAT',
            'test_mode': 'BOOLEAN DEFAULT FALSE'
        }

class ExecutionMeta(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.user_id: Optional[int] = kwargs.get('user_id')
        self.interaction_id: str = kwargs.get('interaction_id', 'default')
        self.workflow_id: str = kwargs.get('workflow_id', '')
        self.workflow_name: str = kwargs.get('workflow_name', '')
        self.interaction_count: int = kwargs.get('interaction_count', 0)
        self.metadata: Optional[Dict] = kwargs.get('metadata', {})

    def get_table_name(self) -> str:
        return "execution_meta"

    def get_schema(self) -> Dict[str, str]:
        return {
            'user_id': 'INTEGER REFERENCES users(id) ON DELETE SET NULL',
            'interaction_id': 'VARCHAR(100) NOT NULL DEFAULT \'default\'',
            'workflow_id': 'VARCHAR(100) NOT NULL',
            'workflow_name': 'VARCHAR(200) NOT NULL',
            'interaction_count': 'INTEGER DEFAULT 0',
            'metadata': 'TEXT'  # JSON string
        }

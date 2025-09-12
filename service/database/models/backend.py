"""
채팅 관련 데이터 모델
"""
from typing import Dict, Optional
from service.database.models.base_model import BaseModel

class BackendLogs(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.user_id: Optional[int] = kwargs.get('user_id')
        self.log_id: str = kwargs.get('log_id', '')
        self.log_level: str = kwargs.get('log_level', 'INFO')
        self.message: str = kwargs.get('message', '')
        self.function_name: Optional[str] = kwargs.get('function_name', '')
        self.api_endpoint: Optional[str] = kwargs.get('api_endpoint', '')
        self.metadata: Optional[Dict] = kwargs.get('metadata', {})

    def get_table_name(self) -> str:
        return "backend_logs"

    def get_schema(self) -> Dict[str, str]:
        return {
            'user_id': 'INTEGER REFERENCES users(id) ON DELETE SET NULL',
            'log_id': 'VARCHAR(200) NOT NULL',
            'log_level': 'VARCHAR(20) DEFAULT \'INFO\'',
            'message': 'TEXT NOT NULL',
            'function_name': 'VARCHAR(200)',
            'api_endpoint': 'VARCHAR(200)',
            'metadata': 'TEXT'
        }

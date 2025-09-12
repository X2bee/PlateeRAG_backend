"""
채팅 관련 데이터 모델
"""
from typing import Dict, Optional
from service.database.models.base_model import BaseModel

class DeployMeta(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.user_id: Optional[int] = kwargs.get('user_id')
        self.workflow_id: str = kwargs.get('workflow_id', '')
        self.workflow_name: str = kwargs.get('workflow_name', '')
        self.is_deployed: bool = kwargs.get('is_deployed', False)
        self.inquire_deploy: bool = kwargs.get('inquire_deploy', False)
        self.is_accepted: bool = kwargs.get('is_accepted', True)
        self.deploy_key: str = kwargs.get('deploy_key', '')

    def get_table_name(self) -> str:
        return "deploy_meta"

    def get_schema(self) -> Dict[str, str]:
        return {
            'user_id': 'INTEGER REFERENCES users(id) ON DELETE SET NULL',
            'workflow_id': 'VARCHAR(100) NOT NULL',
            'workflow_name': 'VARCHAR(200) NOT NULL',
            'is_deployed': 'BOOLEAN DEFAULT FALSE',
            'inquire_deploy': 'BOOLEAN DEFAULT FALSE',
            'is_accepted': 'BOOLEAN DEFAULT TRUE',
            'deploy_key': 'VARCHAR(100)'
        }

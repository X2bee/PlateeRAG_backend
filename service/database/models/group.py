"""
사용자 관련 데이터 모델
"""
from typing import Dict, Optional
from service.database.models.base_model import BaseModel

class GroupMeta(BaseModel):
    """그룹 메타데이터 모델"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.group_name: str = kwargs.get('group_name', "none")
        self.available: bool = kwargs.get('available', True)
        self.available_sections: Optional[list] = kwargs.get('available_sections', [])

    def get_table_name(self) -> str:
        return "group_meta"

    def get_schema(self) -> Dict[str, str]:
        return {
            'group_name': 'VARCHAR(50) UNIQUE NOT NULL',
            'available': 'BOOLEAN DEFAULT TRUE',
            'available_sections': 'TEXT[]',
        }

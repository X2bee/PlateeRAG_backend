"""
채팅 관련 데이터 모델
"""
from typing import Dict, Optional
from service.database.models.base_model import BaseModel
import datetime

class VectorDB(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.user_id: Optional[int] = kwargs.get('user_id')
        self.collection_make_name: str = kwargs.get('collection_make_name', '')
        self.collection_name: str = kwargs.get('collection_name', '')
        self.description: str = kwargs.get('description', '')
        self.registered_at: datetime.datetime = kwargs.get('registered_at', datetime.datetime.now())
        self.updated_at: datetime.datetime = kwargs.get('updated_at', datetime.datetime.now())

    def get_table_name(self) -> str:
        return "vector_db"

    def get_schema(self) -> Dict[str, str]:
        return {
            'user_id': 'INTEGER REFERENCES users(id) ON DELETE SET NULL',
            'collection_make_name': 'VARCHAR(500) NOT NULL',
            'collection_name': 'VARCHAR(500) NOT NULL',
            'description': 'TEXT',
            'registered_at': 'TIMESTAMP',
            'updated_at': 'TIMESTAMP'
        }

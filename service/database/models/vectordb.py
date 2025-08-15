"""
채팅 관련 데이터 모델
"""
from typing import Dict, List, Optional
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

class VectorDBChunkMeta(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.user_id: Optional[int] = kwargs.get('user_id')
        self.collection_name: str = kwargs.get('collection_name', '')
        self.file_name: str = kwargs.get('file_name', '')
        self.chunk_id: str = kwargs.get('chunk_id', None)
        self.chunk_text: str = kwargs.get('chunk_text', '')
        self.chunk_index: str = kwargs.get('chunk_index', '')
        self.total_chunks: int = kwargs.get('total_chunks', 0)
        self.chunk_size: int = kwargs.get('chunk_size', 0)
        self.summary: str = kwargs.get('summary', '')
        self.keywords: List[str] = kwargs.get('keywords', [])
        self.topics: List[str] = kwargs.get('topics', [])
        self.entities: List[str] = kwargs.get('entities', [])
        self.sentiment: str = kwargs.get('sentiment', '')
        self.document_type: str = kwargs.get('document_type', '')
        self.language: str = kwargs.get('language', '')
        self.complexity_level: str = kwargs.get('complexity_level', '')
        self.main_concepts: List[str] = kwargs.get('main_concepts', [])

    def get_table_name(self) -> str:
        return "vector_db_chunk_meta"

    def get_schema(self) -> Dict[str, str]:
        return {
            'user_id': 'INTEGER REFERENCES users(id) ON DELETE SET NULL',
            'collection_name': 'VARCHAR(500) NOT NULL',
            'file_name': 'VARCHAR(500) NOT NULL',
            'chunk_id': 'VARCHAR(500) NOT NULL',
            'chunk_text': 'TEXT NOT NULL',
            'chunk_index': 'INTEGER NOT NULL',
            'total_chunks': 'INTEGER NOT NULL',
            'chunk_size': 'INTEGER NOT NULL',
            'summary': 'TEXT',
            'keywords': 'TEXT[]',
            'topics': 'TEXT[]',
            'entities': 'TEXT[]',
            'sentiment': 'VARCHAR(50)',
            'document_type': 'VARCHAR(50)',
            'language': 'VARCHAR(10)',
            'complexity_level': 'VARCHAR(50)',
            'main_concepts': 'TEXT[]'
        }

class VectorDBChunkEdge(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.user_id: Optional[int] = kwargs.get('user_id')
        self.target: str = kwargs.get('target', '')
        self.source: str = kwargs.get('source', '')
        self.relation_type: str = kwargs.get('relation_type', '')
        self.edge_type: str = kwargs.get('edge_type', 'indirect')
        self.edge_weight: float = kwargs.get('edge_weight', 1.0)

    def get_table_name(self) -> str:
        return "vector_db_chunk_edge"

    def get_schema(self) -> Dict[str, str]:
        return {
            'user_id': 'INTEGER REFERENCES users(id) ON DELETE SET NULL',
            'target': 'VARCHAR(500) NOT NULL',
            'source': 'VARCHAR(500) NOT NULL',
            'relation_type': 'VARCHAR(50) NOT NULL',
            'edge_type': 'VARCHAR(50) NOT NULL',
            'edge_weight': 'FLOAT NOT NULL'
        }

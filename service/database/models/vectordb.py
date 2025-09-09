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
        self.registered_at: datetime.datetime = kwargs.get('registered_at', self.now())
        self.vector_size: int = kwargs.get('vector_size', 0)
        self.init_embedding_model: str = kwargs.get('init_embedding_model', '')
        self.is_shared: bool = kwargs.get('is_shared', False)
        self.share_group: Optional[str] = kwargs.get('share_group', None)
        self.share_permissions: Optional[str] = kwargs.get('share_permissions', 'read')

    def get_table_name(self) -> str:
        return "vector_db"

    def get_schema(self) -> Dict[str, str]:
        return {
            'user_id': 'INTEGER REFERENCES users(id) ON DELETE SET NULL',
            'collection_make_name': 'VARCHAR(500) NOT NULL',
            'collection_name': 'VARCHAR(500) NOT NULL',
            'description': 'TEXT',
            'registered_at': 'TIMESTAMP',
            'vector_size': 'INTEGER DEFAULT 0',
            'init_embedding_model': 'VARCHAR(100)',
            'is_shared': 'BOOLEAN DEFAULT FALSE',
            'share_group': 'VARCHAR(50)',
            'share_permissions': 'VARCHAR(50)'
        }

class VectorDBFolders(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.user_id: Optional[int] = kwargs.get('user_id')
        self.collection_name: str = kwargs.get('collection_name', '')
        self.collection_make_name: str = kwargs.get('collection_make_name', '')
        self.collection_id: int = kwargs.get('collection_id', None)
        self.folder_name: str = kwargs.get('folder_name', '')
        self.parent_folder_name: Optional[str] = kwargs.get('parent_folder_name', None)
        self.parent_folder_id: Optional[int] = kwargs.get('parent_folder_id', None)
        self.is_root: bool = kwargs.get('is_root', False)
        self.full_path: str = kwargs.get('full_path', '')
        self.order_index: int = kwargs.get('order_index', 0)

    def get_table_name(self) -> str:
        return "vector_db_folders"

    def get_schema(self) -> Dict[str, str]:
        return {
            'user_id': 'INTEGER REFERENCES users(id) ON DELETE SET NULL',
            'collection_make_name': 'VARCHAR(500) NOT NULL',
            'collection_name': 'VARCHAR(500) NOT NULL',
            'collection_id': 'INTEGER REFERENCES vector_db(id) ON DELETE CASCADE',
            'folder_name': 'VARCHAR(500) NOT NULL',
            'parent_folder_name': 'VARCHAR(500)',
            'parent_folder_id': 'INTEGER',
            'is_root': 'BOOLEAN DEFAULT FALSE',
            'full_path': 'VARCHAR(1000) NOT NULL',
            'order_index': 'INTEGER DEFAULT 0'
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
        self.document_id: str = kwargs.get('document_id', '')
        self.document_type: str = kwargs.get('document_type', '')
        self.language: str = kwargs.get('language', '')
        self.complexity_level: str = kwargs.get('complexity_level', '')
        self.main_concepts: List[str] = kwargs.get('main_concepts', [])
        self.embedding_provider: Optional[str] = kwargs.get('embedding_provider', 'default')
        self.embedding_model_name: Optional[str] = kwargs.get('embedding_model_name', 'default')
        self.embedding_dimension: Optional[int] = kwargs.get('embedding_dimension', 0)

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
            'document_id': 'VARCHAR(500)',
            'document_type': 'VARCHAR(50)',
            'language': 'VARCHAR(10)',
            'complexity_level': 'VARCHAR(50)',
            'main_concepts': 'TEXT[]',
            'embedding_provider': 'VARCHAR(50)',
            'embedding_model_name': 'VARCHAR(50)',
            'embedding_dimension': 'INTEGER'
        }

class VectorDBChunkEdge(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.user_id: Optional[int] = kwargs.get('user_id')
        self.collection_name: str = kwargs.get('collection_name', '')
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
            'collection_name': 'VARCHAR(500) NOT NULL',
            'target': 'VARCHAR(500) NOT NULL',
            'source': 'VARCHAR(500) NOT NULL',
            'relation_type': 'VARCHAR(50) NOT NULL',
            'edge_type': 'VARCHAR(50) NOT NULL',
            'edge_weight': 'FLOAT NOT NULL'
        }

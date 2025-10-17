
from typing import Dict, Optional
from service.database.models.base_model import BaseModel

class Tools(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.user_id: Optional[int] = kwargs.get('user_id')
        self.function_name: str = kwargs.get('function_name', '')
        self.function_id: str = kwargs.get('function_id', '')
        self.description: str = kwargs.get('description', '')
        self.api_header: Optional[dict] = kwargs.get('api_header', {})
        self.api_body: Optional[dict] = kwargs.get('api_body', {})
        self.api_url: str = kwargs.get('api_url', '')
        self.api_method: str = kwargs.get('api_method', '')
        self.api_timeout: int = kwargs.get('api_timeout', 30)
        self.response_filter: bool = kwargs.get('response_filter', False)
        self.response_filter_path: Optional[str] = kwargs.get('response_filter_path', '')
        self.response_filter_field: Optional[str] = kwargs.get('response_filter_field', '')
        self.status: Optional[str] = kwargs.get('status', '')
        self.is_shared: bool = kwargs.get('is_shared', False)
        self.share_group: Optional[str] = kwargs.get('share_group', None)
        self.share_permissions: Optional[str] = kwargs.get('share_permissions', 'read')
        self.metadata: Optional[Dict] = kwargs.get('metadata', {})

    def get_table_name(self) -> str:
        return "tools"

    def get_schema(self) -> Dict[str, str]:
        return {
            'user_id': 'INTEGER REFERENCES users(id) ON DELETE SET NULL',
            'function_name': 'VARCHAR(100) NOT NULL',
            'function_id': 'VARCHAR(100) NOT NULL',
            'description': 'TEXT',
            'api_header': 'TEXT',
            'api_body': 'TEXT',
            'api_url': 'TEXT NOT NULL',
            'api_method': 'VARCHAR(10) NOT NULL',
            'api_timeout': 'INTEGER DEFAULT 30',
            'response_filter': 'BOOLEAN DEFAULT FALSE',
            'response_filter_path': 'VARCHAR(200)',
            'response_filter_field': 'VARCHAR(100)',
            'status': 'VARCHAR(50)',
            'is_shared': 'BOOLEAN DEFAULT FALSE',
            'share_group': 'VARCHAR(50)',
            'share_permissions': 'VARCHAR(50)',
            'metadata': 'TEXT'
        }

class ToolStoreMeta(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.user_id: Optional[int] = kwargs.get('user_id')
        self.function_upload_id: str = kwargs.get('function_upload_id', '')
        self.function_data: Optional[Dict] = kwargs.get('function_data', {})
        self.metadata: Optional[Dict] = kwargs.get('metadata', {})
        self.rating_count: int = kwargs.get('rating_count', 0)
        self.rating_sum: int = kwargs.get('rating_sum', 0)

    def get_table_name(self) -> str:
        return "tool_store_meta"

    def get_schema(self) -> Dict[str, str]:
        return {
            'user_id': 'INTEGER REFERENCES users(id) ON DELETE SET NULL',
            'function_upload_id': 'VARCHAR(100) NOT NULL',
            'function_data': 'TEXT',
            'metadata': 'TEXT',
            'rating_count': 'INTEGER DEFAULT 0',
            'rating_sum': 'INTEGER DEFAULT 0',
        }

class ToolStoreRating(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.user_id: Optional[int] = kwargs.get('user_id')
        self.tool_store_id: str = kwargs.get('tool_store_id', '')
        self.function_upload_id: str = kwargs.get('function_upload_id', '')
        self.rating: int = kwargs.get('rating', 1)  # 1 to 5

    def get_table_name(self) -> str:
        return "tool_store_rating"

    def get_schema(self) -> Dict[str, str]:
        return {
            'user_id': 'INTEGER REFERENCES users(id) ON DELETE CASCADE',
            'tool_store_id': 'INTEGER REFERENCES tool_store_meta(id) ON DELETE CASCADE',
            'function_upload_id': 'VARCHAR(100) NOT NULL',
            'rating': 'INTEGER CHECK (rating >= 1 AND rating <= 5)'
        }

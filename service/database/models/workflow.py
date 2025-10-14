"""
채팅 관련 데이터 모델
"""
from typing import Dict, Optional, List
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
        self.is_shared: bool = kwargs.get('is_shared', False)
        self.share_group: Optional[str] = kwargs.get('share_group', None)
        self.share_permissions: Optional[str] = kwargs.get('share_permissions', 'read')
        self.current_version: float = kwargs.get('current_version', 1.0)
        self.latest_version: float = kwargs.get('latest_version', 1.0)
        self.workflow_data: Optional[Dict] = kwargs.get('workflow_data', {})

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
            'metadata': 'TEXT',
            'is_shared': 'BOOLEAN DEFAULT FALSE',
            'share_group': 'VARCHAR(50)',
            'share_permissions': 'VARCHAR(50)',
            'current_version': 'FLOAT DEFAULT 1.0',
            'latest_version': 'FLOAT DEFAULT 1.0',
            'workflow_data': 'TEXT'
        }

class WorkflowVersion(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.user_id: Optional[int] = kwargs.get('user_id')
        self.workflow_meta_id: int = kwargs.get('workflow_meta_id')
        self.workflow_id: str = kwargs.get('workflow_id', '')
        self.workflow_name: str = kwargs.get('workflow_name', '')
        self.version: float = kwargs.get('version', 1.0)
        self.version_label: str = kwargs.get('version_label', f"v{self.version}")
        self.current_use: bool = kwargs.get('current_use', False)
        self.workflow_data: Optional[Dict] = kwargs.get('workflow_data', {})

    def get_table_name(self) -> str:
        return "workflow_version"

    def get_schema(self) -> Dict[str, str]:
        return {
            'user_id': 'INTEGER REFERENCES users(id) ON DELETE SET NULL',
            'workflow_meta_id': 'INTEGER REFERENCES workflow_meta(id) ON DELETE CASCADE',
            'workflow_id': 'VARCHAR(100) NOT NULL',
            'workflow_name': 'VARCHAR(200) NOT NULL',
            'version': 'FLOAT DEFAULT 1.0',
            'version_label': 'VARCHAR(50)',
            'current_use': 'BOOLEAN DEFAULT FALSE',
            'workflow_data': 'TEXT'
        }

class WorkflowStoreMeta(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.user_id: Optional[int] = kwargs.get('user_id')
        self.workflow_id: str = kwargs.get('workflow_id', '')
        self.workflow_name: str = kwargs.get('workflow_name', '')
        self.workflow_upload_name: str = kwargs.get('workflow_upload_name', '')
        self.node_count: int = kwargs.get('node_count', 0)
        self.edge_count: int = kwargs.get('edge_count', 0)
        self.has_startnode: bool = kwargs.get('has_startnode', False)
        self.has_endnode: bool = kwargs.get('has_endnode', False)
        self.is_completed: bool = kwargs.get('is_completed', False)
        self.metadata: Optional[Dict] = kwargs.get('metadata', {})
        self.current_version: float = kwargs.get('current_version', 1.0)
        self.latest_version: float = kwargs.get('latest_version', 1.0)
        self.is_template: bool = kwargs.get('is_template', False)
        self.description: Optional[str] = kwargs.get('description', '')
        self.tags: List[str] = kwargs.get('tags', [])
        self.workflow_data: Optional[Dict] = kwargs.get('workflow_data', {})

    def get_table_name(self) -> str:
        return "workflow_store_meta"

    def get_schema(self) -> Dict[str, str]:
        return {
            'user_id': 'INTEGER REFERENCES users(id) ON DELETE SET NULL',
            'workflow_id': 'VARCHAR(100) NOT NULL',
            'workflow_name': 'VARCHAR(200) NOT NULL',
            'workflow_upload_name': 'VARCHAR(200) NOT NULL',
            'node_count': 'INTEGER DEFAULT 0',
            'edge_count': 'INTEGER DEFAULT 0',
            'has_startnode': 'BOOLEAN DEFAULT FALSE',
            'has_endnode': 'BOOLEAN DEFAULT FALSE',
            'is_completed': 'BOOLEAN DEFAULT FALSE',
            'metadata': 'TEXT',
            'current_version': 'FLOAT DEFAULT 1.0',
            'latest_version': 'FLOAT DEFAULT 1.0',
            'is_template': 'BOOLEAN DEFAULT FALSE',
            'description': 'TEXT',
            'tags': 'TEXT[]',
            'workflow_data': 'TEXT'
        }

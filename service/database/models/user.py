"""
사용자 관련 데이터 모델
"""
from typing import Dict, Optional, Union, List
from service.database.models.base_model import BaseModel

class User(BaseModel):
    """사용자 모델"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.username: str = kwargs.get('username', '')
        self.email: str = kwargs.get('email', '')
        self.password_hash: str = kwargs.get('password_hash', '')
        self.full_name: Optional[str] = kwargs.get('full_name')
        self.is_active: bool = kwargs.get('is_active', True)
        self.is_admin: bool = kwargs.get('is_admin', False)
        self.user_type: str = kwargs.get('user_type', "standard")
        self.last_login: Optional[str] = kwargs.get('last_login')
        self.preferences: Optional[Dict] = kwargs.get('preferences', {})
        self.groups: List[str] = kwargs.get('groups', [])
        self.group_name: str = kwargs.get('group_name', 'none')


    def get_table_name(self) -> str:
        return "users"

    def get_schema(self) -> Dict[str, str]:
        return {
            'username': 'VARCHAR(50) UNIQUE NOT NULL',
            'email': 'VARCHAR(100) UNIQUE NOT NULL',
            'password_hash': 'VARCHAR(255) NOT NULL',
            'full_name': 'VARCHAR(100)',
            'is_active': 'BOOLEAN DEFAULT TRUE',
            'is_admin': 'BOOLEAN DEFAULT FALSE',
            'user_type': "VARCHAR(50) DEFAULT 'standard'",
            'groups': 'TEXT[]',
            'group_name': "VARCHAR(50) DEFAULT 'none'",
            'last_login': 'TIMESTAMP',
            'preferences': 'TEXT'
        }

class UserSession(BaseModel):
    """사용자 세션 모델"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.user_id: int = kwargs.get('user_id', 0)
        self.session_token: str = kwargs.get('session_token', '')
        self.expires_at: str = kwargs.get('expires_at', '')
        self.ip_address: Optional[str] = kwargs.get('ip_address')
        self.user_agent: Optional[str] = kwargs.get('user_agent')
        self.is_active: bool = kwargs.get('is_active', True)

    def get_table_name(self) -> str:
        return "user_sessions"

    def get_schema(self) -> Dict[str, str]:
        return {
            'user_id': 'INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE',
            'session_token': 'VARCHAR(255) UNIQUE NOT NULL',
            'expires_at': 'TIMESTAMP NOT NULL',
            'ip_address': 'VARCHAR(45)',
            'user_agent': 'TEXT',
            'is_active': 'BOOLEAN DEFAULT TRUE'
        }

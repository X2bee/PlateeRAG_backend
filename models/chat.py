"""
채팅 관련 데이터 모델
"""
from typing import Dict, Optional
from models.base_model import BaseModel

class ChatSession(BaseModel):
    """채팅 세션 모델"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.user_id: Optional[int] = kwargs.get('user_id')
        self.session_name: str = kwargs.get('session_name', '')
        self.model_used: str = kwargs.get('model_used', '')
        self.total_tokens: int = kwargs.get('total_tokens', 0)
        self.total_cost: float = kwargs.get('total_cost', 0.0)
        self.is_active: bool = kwargs.get('is_active', True)
        self.metadata: Optional[Dict] = kwargs.get('metadata', {})
    
    def get_table_name(self) -> str:
        return "chat_sessions"
    
    def get_schema(self) -> Dict[str, str]:
        return {
            'user_id': 'INTEGER REFERENCES users(id) ON DELETE SET NULL',
            'session_name': 'VARCHAR(200) NOT NULL',
            'model_used': 'VARCHAR(100)',
            'total_tokens': 'INTEGER DEFAULT 0',
            'total_cost': 'DECIMAL(10,4) DEFAULT 0.0',
            'is_active': 'BOOLEAN DEFAULT TRUE',
            'metadata': 'TEXT'  # JSON string
        }

class ChatMessage(BaseModel):
    """채팅 메시지 모델"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.session_id: int = kwargs.get('session_id', 0)
        self.role: str = kwargs.get('role', 'user')  # user, assistant, system
        self.content: str = kwargs.get('content', '')
        self.token_count: int = kwargs.get('token_count', 0)
        self.cost: float = kwargs.get('cost', 0.0)
        self.model_used: str = kwargs.get('model_used', '')
        self.response_time: Optional[float] = kwargs.get('response_time')
        self.metadata: Optional[Dict] = kwargs.get('metadata', {})
    
    def get_table_name(self) -> str:
        return "chat_messages"
    
    def get_schema(self) -> Dict[str, str]:
        return {
            'session_id': 'INTEGER NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE',
            'role': 'VARCHAR(20) NOT NULL',
            'content': 'TEXT NOT NULL',
            'token_count': 'INTEGER DEFAULT 0',
            'cost': 'DECIMAL(10,6) DEFAULT 0.0',
            'model_used': 'VARCHAR(100)',
            'response_time': 'DECIMAL(8,3)',  # seconds
            'metadata': 'TEXT'  # JSON string
        }

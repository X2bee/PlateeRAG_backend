"""
DB Sync Configuration 모델
외부 DB에서 주기적으로 데이터를 가져오는 설정
"""
from typing import Dict, Optional
from service.database.models.base_model import BaseModel

class DBSyncConfig(BaseModel):
    """DB 자동 동기화 설정 모델"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # 기본 식별자
        self.manager_id: str = kwargs.get('manager_id', '')
        self.user_id: str = kwargs.get('user_id', '')
        
        # ✨ DB 연결 정보 (암호화된 비밀번호)
        self.db_type: str = kwargs.get('db_type', 'postgresql')  # postgresql, mysql, sqlite
        self.db_host: Optional[str] = kwargs.get('db_host')
        self.db_port: Optional[int] = kwargs.get('db_port')
        self.db_name: str = kwargs.get('db_name', '')
        self.db_username: Optional[str] = kwargs.get('db_username')
        self.db_password: Optional[str] = kwargs.get('db_password')  # 암호화된 비밀번호
        
        # 동기화 쿼리/테이블
        self.query: Optional[str] = kwargs.get('query')
        self.table_name: Optional[str] = kwargs.get('table_name')
        self.schema_name: Optional[str] = kwargs.get('schema_name')
        self.chunk_size: Optional[int] = kwargs.get('chunk_size')
        
        # 스케줄 설정
        self.enabled: bool = kwargs.get('enabled', True)
        self.schedule_type: str = kwargs.get('schedule_type', 'interval')
        self.interval_minutes: Optional[int] = kwargs.get('interval_minutes')
        self.cron_expression: Optional[str] = kwargs.get('cron_expression')
        
        # 동기화 옵션
        self.detect_changes: bool = kwargs.get('detect_changes', True)
        self.notification_enabled: bool = kwargs.get('notification_enabled', False)
        
        # 실행 이력
        self.last_sync_at: Optional[str] = kwargs.get('last_sync_at')
        self.last_sync_status: Optional[str] = kwargs.get('last_sync_status')
        self.sync_count: int = kwargs.get('sync_count', 0)
        self.last_checksum: Optional[str] = kwargs.get('last_checksum')
        self.last_error: Optional[str] = kwargs.get('last_error')

    def get_table_name(self) -> str:
        return "db_sync_configs"

    def get_schema(self) -> Dict[str, str]:
        return {
            'manager_id': 'VARCHAR(50) NOT NULL UNIQUE',
            'user_id': 'VARCHAR(50) NOT NULL',
            
            # DB 연결 정보
            'db_type': 'VARCHAR(20) NOT NULL',
            'db_host': 'VARCHAR(255)',
            'db_port': 'INTEGER',
            'db_name': 'VARCHAR(255) NOT NULL',
            'db_username': 'VARCHAR(255)',
            'db_password': 'TEXT',  # 암호화된 비밀번호
            
            # 쿼리/테이블
            'query': 'TEXT',
            'table_name': 'VARCHAR(255)',
            'schema_name': 'VARCHAR(255)',
            'chunk_size': 'INTEGER',
            
            # 스케줄
            'enabled': 'BOOLEAN DEFAULT TRUE',
            'schedule_type': 'VARCHAR(20) NOT NULL',
            'interval_minutes': 'INTEGER',
            'cron_expression': 'VARCHAR(100)',
            
            # 옵션
            'detect_changes': 'BOOLEAN DEFAULT TRUE',
            'notification_enabled': 'BOOLEAN DEFAULT FALSE',
            
            # 이력
            'last_sync_at': 'TIMESTAMP',
            'last_sync_status': 'VARCHAR(50)',
            'sync_count': 'INTEGER DEFAULT 0',
            'last_checksum': 'VARCHAR(64)',
            'last_error': 'TEXT'
        }

"""
DB Sync Configuration 모델
외부 DB에서 주기적으로 데이터를 가져오는 설정
"""
from typing import Dict, Optional
from datetime import datetime
from service.database.models.base_model import BaseModel


class DBSyncConfig(BaseModel):
    """DB 자동 동기화 설정 모델"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # 기본 식별자
        self.manager_id: str = kwargs.get('manager_id', '')
        self.user_id: str = kwargs.get('user_id', '')
        
        # DB 연결 정보
        self.db_type: str = kwargs.get('db_type', 'postgresql')
        self.db_host: Optional[str] = kwargs.get('db_host')
        self.db_port: Optional[int] = kwargs.get('db_port')
        self.db_name: str = kwargs.get('db_name', '')
        self.db_username: Optional[str] = kwargs.get('db_username')
        self.db_password: Optional[str] = kwargs.get('db_password')  # 암호화되어 저장됨
        
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
        
        # MLflow 자동 업로드 설정
        self.mlflow_enabled: bool = kwargs.get('mlflow_enabled', False)
        self.mlflow_experiment_name: Optional[str] = kwargs.get('mlflow_experiment_name')
        self.mlflow_tracking_uri: Optional[str] = kwargs.get('mlflow_tracking_uri')
        self.mlflow_upload_count: int = kwargs.get('mlflow_upload_count', 0)
        
        # 실행 이력
        self.last_sync_at: Optional[str] = kwargs.get('last_sync_at')
        self.last_sync_status: Optional[str] = kwargs.get('last_sync_status')
        self.sync_count: int = kwargs.get('sync_count', 0)
        self.last_checksum: Optional[str] = kwargs.get('last_checksum')
        self.last_error: Optional[str] = kwargs.get('last_error')
        
        # MLflow 업로드 이력
        self.last_mlflow_upload_at: Optional[str] = kwargs.get('last_mlflow_upload_at')
        self.last_mlflow_run_id: Optional[str] = kwargs.get('last_mlflow_run_id')
        self.last_mlflow_error: Optional[str] = kwargs.get('last_mlflow_error')
        
        # 타임스탬프
        self.created_at: Optional[str] = kwargs.get('created_at')
        self.updated_at: Optional[str] = kwargs.get('updated_at')

    def get_table_name(self) -> str:
        """테이블 이름 반환"""
        return "db_sync_configs"

    def get_schema(self) -> Dict[str, str]:
        """테이블 스키마 정의"""
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
            
            # MLflow 설정
            'mlflow_enabled': 'BOOLEAN DEFAULT FALSE',
            'mlflow_experiment_name': 'VARCHAR(255)',
            'mlflow_tracking_uri': 'VARCHAR(500)',
            'mlflow_upload_count': 'INTEGER DEFAULT 0',
            
            # 이력
            'last_sync_at': 'TIMESTAMP',
            'last_sync_status': 'VARCHAR(50)',
            'sync_count': 'INTEGER DEFAULT 0',
            'last_checksum': 'VARCHAR(64)',
            'last_error': 'TEXT',
            
            # MLflow 업로드 이력
            'last_mlflow_upload_at': 'TIMESTAMP',
            'last_mlflow_run_id': 'VARCHAR(100)',
            'last_mlflow_error': 'TEXT',
            
            # 타임스탬프
            'created_at': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP',
            'updated_at': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP'
        }
    
    # ==================== 상태 확인 ====================
    
    def is_active(self) -> bool:
        """활성화 상태 확인"""
        return self.enabled
    
    def get_schedule_description(self) -> str:
        """스케줄 설명 반환"""
        if self.schedule_type == 'interval':
            minutes = self.interval_minutes or 60
            if minutes < 60:
                return f"매 {minutes}분"
            elif minutes == 60:
                return "매 시간"
            elif minutes % 60 == 0:
                hours = minutes // 60
                return f"매 {hours}시간"
            else:
                hours = minutes // 60
                mins = minutes % 60
                return f"매 {hours}시간 {mins}분"
        elif self.schedule_type == 'cron':
            return f"크론: {self.cron_expression or '설정 없음'}"
        else:
            return "알 수 없는 스케줄"
    
    def get_last_sync_info(self) -> Dict:
        """마지막 동기화 정보 반환"""
        return {
            'last_sync_at': self.last_sync_at,
            'last_sync_status': self.last_sync_status,
            'sync_count': self.sync_count,
            'last_error': self.last_error,
            'last_checksum': self.last_checksum
        }
    
    # ==================== MLflow 관련 ====================
    
    def get_next_mlflow_dataset_name(self) -> str:
        """다음 MLflow 데이터셋 이름 생성 (버전 포함)"""
        if not self.mlflow_experiment_name:
            return f"dataset_v{self.mlflow_upload_count + 1}"
        return f"{self.mlflow_experiment_name}_v{self.mlflow_upload_count + 1}"
    
    def get_mlflow_info(self) -> Dict:
        """MLflow 설정 및 상태 정보 반환"""
        return {
            'mlflow_enabled': self.mlflow_enabled,
            'mlflow_experiment_name': self.mlflow_experiment_name,
            'mlflow_tracking_uri': self.mlflow_tracking_uri,
            'mlflow_upload_count': self.mlflow_upload_count,
            'last_mlflow_upload_at': self.last_mlflow_upload_at,
            'last_mlflow_run_id': self.last_mlflow_run_id,
            'last_mlflow_error': self.last_mlflow_error,
            'next_dataset_name': (
                self.get_next_mlflow_dataset_name() 
                if self.mlflow_enabled 
                else None
            )
        }
    
    # ==================== 상태 업데이트 헬퍼 (Deprecated) ====================
    # Note: 이 메서드들은 하위 호환성을 위해 유지되지만,
    # 새 코드에서는 직접 필드를 수정하고 updated_at을 설정하는 것을 권장
    
    def mark_sync_success(self, checksum: Optional[str] = None):
        """
        동기화 성공 기록 (Deprecated)
        
        Note: 새 코드에서는 직접 필드를 업데이트하세요:
            config.last_sync_at = datetime.now().isoformat()
            config.last_sync_status = 'success'
            config.sync_count += 1
            config.updated_at = datetime.now().isoformat()
        """
        self.last_sync_at = datetime.now().isoformat()
        self.last_sync_status = 'success'
        self.sync_count += 1
        if checksum:
            self.last_checksum = checksum
        self.last_error = None
        self.updated_at = datetime.now().isoformat()
    
    def mark_sync_failed(self, error_message: str):
        """
        동기화 실패 기록 (Deprecated)
        
        Note: 새 코드에서는 직접 필드를 업데이트하세요:
            config.last_sync_at = datetime.now().isoformat()
            config.last_sync_status = 'failed'
            config.last_error = error_message[:500]
            config.updated_at = datetime.now().isoformat()
        """
        self.last_sync_at = datetime.now().isoformat()
        self.last_sync_status = 'failed'
        self.last_error = error_message[:500]
        self.updated_at = datetime.now().isoformat()
    
    def mark_no_changes(self):
        """
        변경사항 없음 기록 (Deprecated)
        
        Note: 새 코드에서는 직접 필드를 업데이트하세요:
            config.last_sync_at = datetime.now().isoformat()
            config.last_sync_status = 'no_changes'
            config.updated_at = datetime.now().isoformat()
        """
        self.last_sync_at = datetime.now().isoformat()
        self.last_sync_status = 'no_changes'
        self.updated_at = datetime.now().isoformat()
    
    def mark_mlflow_upload_success(self, run_id: str):
        """
        MLflow 업로드 성공 기록 (Deprecated)
        
        Note: 새 코드에서는 직접 필드를 업데이트하세요:
            config.last_mlflow_upload_at = datetime.now().isoformat()
            config.last_mlflow_run_id = run_id
            config.mlflow_upload_count += 1
            config.last_mlflow_error = None
            config.updated_at = datetime.now().isoformat()
        """
        self.last_mlflow_upload_at = datetime.now().isoformat()
        self.last_mlflow_run_id = run_id
        self.mlflow_upload_count += 1
        self.last_mlflow_error = None
        self.updated_at = datetime.now().isoformat()
    
    def mark_mlflow_upload_failed(self, error_message: str):
        """
        MLflow 업로드 실패 기록 (Deprecated)
        
        Note: 새 코드에서는 직접 필드를 업데이트하세요:
            config.last_mlflow_upload_at = datetime.now().isoformat()
            config.last_mlflow_error = error_message[:500]
            config.updated_at = datetime.now().isoformat()
        """
        self.last_mlflow_upload_at = datetime.now().isoformat()
        self.last_mlflow_error = error_message[:500]
        self.updated_at = datetime.now().isoformat()
    
    # ==================== DB 연결 정보 ====================
    
    def get_db_connection_string(self, include_password: bool = False) -> str:
        """
        DB 연결 문자열 생성 (로깅/디버깅용)
        
        Args:
            include_password: 비밀번호 포함 여부 (기본값: False)
        """
        password_display = '***' if self.db_password else 'N/A'
        if include_password and self.db_password:
            password_display = self.db_password
        
        if self.db_port:
            return (
                f"{self.db_type}://{self.db_username}:{password_display}"
                f"@{self.db_host}:{self.db_port}/{self.db_name}"
            )
        else:
            return (
                f"{self.db_type}://{self.db_username}:{password_display}"
                f"@{self.db_host}/{self.db_name}"
            )
    
    def get_query_summary(self) -> str:
        """쿼리 또는 테이블 정보 요약"""
        if self.query:
            # 쿼리를 50자로 제한
            query_preview = self.query.replace('\n', ' ').strip()
            if len(query_preview) > 50:
                query_preview = query_preview[:47] + "..."
            return f"Query: {query_preview}"
        elif self.table_name:
            if self.schema_name:
                return f"Table: {self.schema_name}.{self.table_name}"
            return f"Table: {self.table_name}"
        else:
            return "N/A"
    
    # ==================== 유효성 검증 ====================
    
    def validate(self) -> tuple[bool, Optional[str]]:
        """
        설정 유효성 검증
        
        Returns:
            (is_valid, error_message) 튜플
        """
        # 필수 필드 확인
        if not self.manager_id:
            return False, "manager_id가 필요합니다"
        
        if not self.user_id:
            return False, "user_id가 필요합니다"
        
        if not self.db_name:
            return False, "db_name이 필요합니다"
        
        # 스케줄 설정 확인
        if self.schedule_type == 'interval':
            if not self.interval_minutes or self.interval_minutes <= 0:
                return False, "interval_minutes는 양수여야 합니다"
        elif self.schedule_type == 'cron':
            if not self.cron_expression:
                return False, "cron_expression이 필요합니다"
        else:
            return False, f"지원되지 않는 schedule_type: {self.schedule_type}"
        
        # 쿼리 또는 테이블 중 하나는 있어야 함
        if not self.query and not self.table_name:
            return False, "query 또는 table_name 중 하나는 필수입니다"
        
        # MLflow 설정 확인
        if self.mlflow_enabled and not self.mlflow_experiment_name:
            return False, "MLflow가 활성화된 경우 experiment_name이 필요합니다"
        
        return True, None
    
    # ==================== 문자열 표현 ====================
    
    def __repr__(self) -> str:
        """객체 문자열 표현"""
        mlflow_status = "MLflow:ON" if self.mlflow_enabled else "MLflow:OFF"
        status = "ENABLED" if self.enabled else "DISABLED"
        
        return (
            f"<DBSyncConfig("
            f"manager={self.manager_id}, "
            f"user={self.user_id}, "
            f"status={status}, "
            f"db={self.db_type}, "
            f"schedule={self.get_schedule_description()}, "
            f"syncs={self.sync_count}, "
            f"{mlflow_status}"
            f")>"
        )
    
    def __str__(self) -> str:
        """사용자 친화적 문자열 표현"""
        return (
            f"DB Sync: {self.manager_id} "
            f"({'활성' if self.enabled else '비활성'}) - "
            f"{self.get_schedule_description()}"
        )
    
    def to_summary_dict(self) -> Dict:
        """요약 정보를 딕셔너리로 반환"""
        return {
            'manager_id': self.manager_id,
            'user_id': self.user_id,
            'enabled': self.enabled,
            'db_type': self.db_type,
            'db_host': self.db_host,
            'db_name': self.db_name,
            'schedule': self.get_schedule_description(),
            'query_summary': self.get_query_summary(),
            'sync_count': self.sync_count,
            'last_sync_status': self.last_sync_status,
            'last_sync_at': self.last_sync_at,
            'mlflow_enabled': self.mlflow_enabled,
            'mlflow_upload_count': self.mlflow_upload_count if self.mlflow_enabled else None,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }
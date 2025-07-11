"""
데이터베이스 연결 및 마이그레이션 관리
"""
import os
import logging
import sqlite3
from typing import Optional, Dict, Any, Union
from pathlib import Path

logger = logging.getLogger("database-manager")

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    POSTGRES_AVAILABLE = True
except ImportError:
    logger.warning("psycopg2 not available, PostgreSQL support disabled")
    POSTGRES_AVAILABLE = False

class DatabaseManager:
    """데이터베이스 연결 및 마이그레이션 관리"""
    
    def __init__(self, database_config=None):
        self.config = database_config
        self.connection = None
        self.db_type = None
        self.logger = logger
    
    def determine_database_type(self) -> str:
        """사용할 데이터베이스 타입 결정"""
        if not self.config:
            return "sqlite"
        
        # DATABASE_TYPE이 명시적으로 설정된 경우
        db_type = self.config.DATABASE_TYPE.value.lower()
        if db_type in ["sqlite", "postgresql"]:
            return db_type
        
        # auto 모드: PostgreSQL 접속 정보 확인
        if db_type == "auto":
            postgres_required_fields = [
                self.config.POSTGRES_HOST.value,
                self.config.POSTGRES_USER.value,
                self.config.POSTGRES_PASSWORD.value
            ]
            
            # 모든 필수 PostgreSQL 정보가 있고 psycopg2가 사용 가능한 경우
            if all(field.strip() for field in postgres_required_fields) and POSTGRES_AVAILABLE:
                self.logger.info("PostgreSQL configuration detected, using PostgreSQL")
                return "postgresql"
            else:
                self.logger.info("Using SQLite as default database")
                return "sqlite"
        
        return "sqlite"
    
    def get_connection_string(self) -> str:
        """데이터베이스 연결 문자열 생성"""
        if self.db_type == "postgresql":
            host = self.config.POSTGRES_HOST.value
            port = self.config.POSTGRES_PORT.value
            database = self.config.POSTGRES_DB.value
            user = self.config.POSTGRES_USER.value
            password = self.config.POSTGRES_PASSWORD.value
            
            return f"postgresql://{user}:{password}@{host}:{port}/{database}"
        
        elif self.db_type == "sqlite":
            sqlite_path = self.config.SQLITE_PATH.value if self.config else "constants/config.db"
            # 디렉토리 생성
            os.makedirs(os.path.dirname(sqlite_path), exist_ok=True)
            return f"sqlite:///{sqlite_path}"
        
        raise ValueError(f"Unsupported database type: {self.db_type}")
    
    def connect(self) -> bool:
        """데이터베이스 연결"""
        try:
            self.db_type = self.determine_database_type()
            
            if self.db_type == "postgresql":
                return self._connect_postgresql()
            elif self.db_type == "sqlite":
                return self._connect_sqlite()
            
        except Exception as e:
            self.logger.error(f"Failed to connect to database: {e}")
            return False
    
    def _connect_postgresql(self) -> bool:
        """PostgreSQL 연결"""
        try:
            self.connection = psycopg2.connect(
                host=self.config.POSTGRES_HOST.value,
                port=self.config.POSTGRES_PORT.value,
                database=self.config.POSTGRES_DB.value,
                user=self.config.POSTGRES_USER.value,
                password=self.config.POSTGRES_PASSWORD.value,
                cursor_factory=RealDictCursor
            )
            self.connection.autocommit = True
            self.logger.info("Successfully connected to PostgreSQL")
            return True
        except Exception as e:
            self.logger.error(f"PostgreSQL connection failed: {e}")
            return False
    
    def _connect_sqlite(self) -> bool:
        """SQLite 연결"""
        try:
            sqlite_path = self.config.SQLITE_PATH.value if self.config else "constants/config.db"
            
            # 디렉토리 생성
            os.makedirs(os.path.dirname(sqlite_path), exist_ok=True)
            
            self.connection = sqlite3.connect(sqlite_path)
            self.connection.row_factory = sqlite3.Row  # 딕셔너리 형태로 결과 반환
            self.logger.info(f"Successfully connected to SQLite: {sqlite_path}")
            return True
        except Exception as e:
            self.logger.error(f"SQLite connection failed: {e}")
            return False
    
    def disconnect(self):
        """데이터베이스 연결 해제"""
        if self.connection:
            self.connection.close()
            self.connection = None
            self.logger.info("Database connection closed")
    
    def execute_query(self, query: str, params: tuple = None) -> Optional[list]:
        """쿼리 실행"""
        if not self.connection:
            self.logger.error("No database connection available")
            return None
        
        try:
            cursor = self.connection.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            # SELECT 쿼리인 경우 결과 반환
            if query.strip().upper().startswith('SELECT'):
                result = cursor.fetchall()
                return [dict(row) for row in result]
            else:
                if self.db_type == "sqlite":
                    self.connection.commit()
                return []
                
        except Exception as e:
            self.logger.error(f"Query execution failed: {e}")
            if self.db_type == "sqlite":
                self.connection.rollback()
            return None
    
    def table_exists(self, table_name: str) -> bool:
        """테이블 존재 여부 확인"""
        if self.db_type == "postgresql":
            query = """
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = %s
                );
            """
            result = self.execute_query(query, (table_name,))
        else:  # sqlite
            query = """
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name=?;
            """
            result = self.execute_query(query, (table_name,))
        
        return bool(result)
    
    def create_config_table(self) -> bool:
        """설정 테이블 생성"""
        if self.db_type == "postgresql":
            query = """
                CREATE TABLE IF NOT EXISTS persistent_configs (
                    id SERIAL PRIMARY KEY,
                    config_path VARCHAR(255) UNIQUE NOT NULL,
                    config_value TEXT,
                    data_type VARCHAR(50) DEFAULT 'string',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """
        else:  # sqlite
            query = """
                CREATE TABLE IF NOT EXISTS persistent_configs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    config_path TEXT UNIQUE NOT NULL,
                    config_value TEXT,
                    data_type TEXT DEFAULT 'string',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """
        
        result = self.execute_query(query)
        if result is not None:
            self.logger.info("Config table created successfully")
            return True
        return False
    
    def run_migrations(self) -> bool:
        """데이터베이스 마이그레이션 실행"""
        try:
            self.logger.info(f"Running migrations for {self.db_type}")
            
            # 1. 기본 설정 테이블 생성
            if not self.create_config_table():
                return False
            
            # 2. 추가 마이그레이션 (향후 확장 가능)
            migrations = [
                self._migration_001_add_indexes,
                # 향후 마이그레이션 함수들 추가
            ]
            
            for migration in migrations:
                if not migration():
                    self.logger.error(f"Migration failed: {migration.__name__}")
                    return False
            
            self.logger.info("All migrations completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Migration failed: {e}")
            return False
    
    def _migration_001_add_indexes(self) -> bool:
        """마이그레이션 001: 인덱스 추가"""
        try:
            if self.db_type == "postgresql":
                query = """
                    CREATE INDEX IF NOT EXISTS idx_config_path 
                    ON persistent_configs(config_path);
                """
            else:  # sqlite
                query = """
                    CREATE INDEX IF NOT EXISTS idx_config_path 
                    ON persistent_configs(config_path);
                """
            
            result = self.execute_query(query)
            return result is not None
        except Exception as e:
            self.logger.error(f"Failed to create indexes: {e}")
            return False

# 전역 데이터베이스 매니저 인스턴스
_db_manager = None

def get_database_manager(database_config=None) -> DatabaseManager:
    """데이터베이스 매니저 싱글톤 인스턴스 반환"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager(database_config)
    return _db_manager

def initialize_database(database_config=None) -> bool:
    """데이터베이스 초기화 및 마이그레이션"""
    db_manager = get_database_manager(database_config)
    
    if not db_manager.connect():
        return False
    
    if database_config and database_config.AUTO_MIGRATION.value:
        return db_manager.run_migrations()
    
    return True

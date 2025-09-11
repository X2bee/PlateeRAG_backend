"""
데이터베이스 연결 및 마이그레이션 관리
"""
import os
import logging
import sqlite3
from typing import Optional, Dict, Any, Union
from pathlib import Path
from zoneinfo import ZoneInfo
logger = logging.getLogger("database-manager")

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    POSTGRES_AVAILABLE = True
except ImportError:
    logger.warning("psycopg2 not available, PostgreSQL support disabled")
    POSTGRES_AVAILABLE = False

TIMEZONE = ZoneInfo(os.getenv('TIMEZONE', 'Asia/Seoul'))

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
        # db_type이 설정되지 않았으면 먼저 결정
        if not self.db_type:
            self.db_type = self.determine_database_type()

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
            self.connection.autocommit = False

            cursor = self.connection.cursor()
            timezone_str = str(TIMEZONE)
            cursor.execute(f"SET timezone = '{timezone_str}'")
            self.connection.commit()
            self.logger.warning("PostgreSQL 세션 타임존을 %s로 설정했습니다 (BaseModel TIMEZONE 사용)", timezone_str)

            self.logger.info("Successfully connected to PostgreSQL")
            return True
        except psycopg2.Error as e:
            self.logger.error("PostgreSQL connection failed: %s", e)
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
                # INSERT, UPDATE, DELETE 등의 경우 commit 필요
                self.connection.commit()
                return []

        except psycopg2.Error as e:
            self.logger.error("PostgreSQL query execution failed: %s", e)
            try:
                self.connection.rollback()
            except psycopg2.Error:
                pass
            return None
        except sqlite3.Error as e:
            self.logger.error("SQLite query execution failed: %s", e)
            try:
                self.connection.rollback()
            except sqlite3.Error:
                pass
            return None
        except Exception as e:
            self.logger.error("Query execution failed: %s", e)
            try:
                self.connection.rollback()
            except:
                pass
            return None

    def execute_query_one(self, query: str, params: tuple = None) -> Optional[Dict]:
        """쿼리 실행하여 단일 결과 반환"""
        result = self.execute_query(query, params)
        if result and len(result) > 0:
            return result[0]
        return None

    def execute_insert(self, query: str, params: tuple = None) -> Optional[int]:
        """INSERT 쿼리 실행하여 생성된 ID 반환"""
        if not self.connection:
            self.logger.error("No database connection available")
            return None

        try:
            cursor = self.connection.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)

            if self.db_type == "sqlite":
                insert_id = cursor.lastrowid
                self.connection.commit()
                return insert_id
            else:  # postgresql
                # PostgreSQL의 경우 RETURNING id를 쿼리에 포함해야 함
                result = cursor.fetchone()
                self.connection.commit()
                if result:
                    return result[0]  # id 값
                return None

        except psycopg2.Error as e:
            self.logger.error("PostgreSQL insert query execution failed: %s", e)
            try:
                self.connection.rollback()
            except psycopg2.Error:
                pass
            return None
        except sqlite3.Error as e:
            self.logger.error("SQLite insert query execution failed: %s", e)
            try:
                self.connection.rollback()
            except sqlite3.Error:
                pass
            return None

    def execute_update_delete(self, query: str, params: tuple = None) -> Optional[int]:
        """UPDATE/DELETE 쿼리 실행하여 영향받은 행 수 반환"""
        if not self.connection:
            self.logger.error("No database connection available")
            return None

        try:
            cursor = self.connection.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)

            affected_rows = cursor.rowcount
            self.connection.commit()
            return affected_rows

        except psycopg2.Error as e:
            self.logger.error("PostgreSQL update/delete query execution failed: %s", e)
            try:
                self.connection.rollback()
            except psycopg2.Error:
                pass
            return None
        except sqlite3.Error as e:
            self.logger.error("SQLite update/delete query execution failed: %s", e)
            try:
                self.connection.rollback()
            except sqlite3.Error:
                pass
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

    def run_migrations(self, models_registry=None) -> bool:
        """데이터베이스 마이그레이션 실행"""
        try:
            self.logger.info("Running migrations for %s", self.db_type)

            # 기존 고정 마이그레이션들
            fixed_migrations = [
                self._migration_001_add_indexes,
                # 향후 고정 마이그레이션 함수들 추가
            ]

            for migration in fixed_migrations:
                if not migration():
                    self.logger.error("Fixed migration failed: %s", migration.__name__)
                    return False

            # 동적 스키마 마이그레이션
            if models_registry:
                if not self._run_schema_migrations(models_registry):
                    self.logger.error("Schema migrations failed")
                    return False

            self.logger.info("All migrations completed successfully")
            return True

        except (sqlite3.Error, ImportError, AttributeError) as e:
            self.logger.error("Migration failed: %s", e)
            return False

    def _run_schema_migrations(self, models_registry) -> bool:
        """스키마 변경 감지 및 마이그레이션 실행"""
        try:
            self.logger.info("Running schema migrations...")
            self.logger.debug(f"Registered models: {[model.__name__ for model in models_registry]}")

            for model_class in models_registry:
                table_name = model_class().get_table_name()
                expected_schema = model_class().get_schema()

                self.logger.debug(f"Checking schema for table: {table_name}")
                self.logger.debug(f"Expected schema: {expected_schema}")

                # Get current table structure
                current_columns = self._get_table_columns(table_name)
                self.logger.debug(f"Current columns: {current_columns}")

                if not current_columns:
                    self.logger.warning(f"Table {table_name} does not exist or has no columns")
                    continue

                # Compare schemas and detect changes
                missing_columns = []
                for column_name, column_def in expected_schema.items():
                    if column_name not in current_columns:
                        missing_columns.append((column_name, column_def))
                        self.logger.info(f"Found missing column: {column_name} ({column_def})")

                self.logger.info(f"Missing columns for {table_name}: {missing_columns}")

                # Add missing columns
                for column_name, column_def in missing_columns:
                    if not self._add_column_to_table(table_name, column_name, column_def):
                        return False

                if not missing_columns:
                    self.logger.info(f"Table {table_name} schema is up to date")

            return True

        except Exception as e:
            self.logger.error(f"Schema migration failed: {e}")
            import traceback
            self.logger.error(f"Schema migration traceback: {traceback.format_exc()}")
            return False

    def _get_table_columns(self, table_name: str) -> dict:
        """테이블의 현재 컬럼 구조 조회"""
        try:
            if self.db_type == "postgresql":
                query = """
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns
                WHERE table_name = %s
                ORDER BY ordinal_position
                """
                result = self.execute_query(query, (table_name,))

                if result:
                    return {row['column_name']: row['data_type'] for row in result}
                else:
                    return {}

            else:  # SQLite
                query = f"PRAGMA table_info({table_name})"
                result = self.execute_query(query)

                if result:
                    # SQLite PRAGMA returns: cid, name, type, notnull, dflt_value, pk
                    return {row['name']: row['type'] for row in result}
                else:
                    return {}

        except Exception as e:
            self.logger.error(f"Failed to get table columns for {table_name}: {e}")
            return {}

    def _add_column_to_table(self, table_name: str, column_name: str, column_def: str) -> bool:
        """테이블에 컬럼 추가"""
        try:
            self.logger.info(f"Adding missing column {column_name} to table {table_name}")

            if self.db_type == "postgresql":
                alter_query = f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS {column_name} {column_def}"
            else:  # SQLite
                alter_query = f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_def}"

            self.logger.info(f"Executing query: {alter_query}")
            result = self.execute_query(alter_query)

            if result is not None:  # None means error, empty list means success
                self.logger.info(f"Successfully added column {column_name} to {table_name}")
                return True
            else:
                self.logger.error(f"Failed to add column {column_name} to {table_name}")
                return False

        except Exception as e:
            self.logger.error(f"Failed to add column {column_name} to {table_name}: {e}")
            return False

    def _migration_001_add_indexes(self) -> bool:
        """마이그레이션 001: 인덱스 추가"""
        try:
            # 이제 persistent_configs 테이블은 모델에서 인덱스와 함께 생성되므로
            # 여기서는 아무것도 하지 않음
            return True
        except (sqlite3.Error, ImportError, AttributeError) as e:
            self.logger.error("Failed to create indexes: %s", e)
            return False

_db_manager = None

def get_database_manager(database_config=None) -> DatabaseManager:
    """데이터베이스 매니저 싱글톤 인스턴스 반환"""
    global _db_manager
    if _db_manager is None or database_config is not None:
        # database_config가 제공되면 새로운 인스턴스 생성
        _db_manager = DatabaseManager(database_config)
    return _db_manager

def reset_database_manager():
    """데이터베이스 매니저 싱글톤 리셋 (테스트용)"""
    global _db_manager
    if _db_manager and _db_manager.connection:
        _db_manager.disconnect()
    _db_manager = None

def initialize_database(database_config=None) -> bool:
    """데이터베이스 초기화 및 마이그레이션"""
    db_manager = get_database_manager(database_config)

    if not db_manager.connect():
        return False

    if database_config and database_config.AUTO_MIGRATION.value:
        return db_manager.run_migrations()

    return True

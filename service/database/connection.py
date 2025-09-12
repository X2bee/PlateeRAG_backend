import logging
import json
from typing import List, Dict, Any, Optional, Type
from service.database.database_manager import DatabaseManager
from service.database.models.base_model import BaseModel

logger = logging.getLogger("app-database")

class AppDatabaseManager:
    def __init__(self, database_config=None):
        self.config_db_manager = DatabaseManager(database_config)
        self.logger = logger
        self._models_registry: List[Type[BaseModel]] = []

    def register_model(self, model_class: Type[BaseModel]):
        """모델 클래스를 등록"""
        if model_class not in self._models_registry:
            self._models_registry.append(model_class)
            self.logger.info("Registered model: %s", model_class.__name__)

    def register_models(self, model_classes: List[Type[BaseModel]]):
        """여러 모델 클래스를 한 번에 등록"""
        for model_class in model_classes:
            self.register_model(model_class)

    def initialize_database(self) -> bool:
        """데이터베이스 연결 및 테이블 생성"""
        try:
            # 기존 config DB 매니저를 사용하여 연결
            if not self.config_db_manager.connect():
                self.logger.error("Failed to connect to database")
                return False

            self.logger.info("Connected to database for application data")

            # 등록된 모델들의 테이블 생성
            return self.create_tables()

        except Exception as e:
            self.logger.error("Failed to initialize application database: %s", e)
            return False

    def create_tables(self) -> bool:
        """등록된 모든 모델의 테이블 생성"""
        try:
            db_type = self.config_db_manager.db_type

            for model_class in self._models_registry:
                table_name = model_class().get_table_name()
                create_query = model_class.get_create_table_query(db_type)

                self.logger.info("Creating table: %s", table_name)
                self.config_db_manager.execute_query(create_query)

                # PersistentConfigModel의 경우 인덱스도 생성
                if hasattr(model_class, '__name__') and model_class.__name__ == 'PersistentConfigModel':
                    index_query = "CREATE INDEX IF NOT EXISTS idx_config_path ON persistent_configs(config_path)"
                    try:
                        self.config_db_manager.execute_query(index_query)
                        self.logger.info("Created index for table: %s", table_name)
                    except (ImportError, AttributeError, ValueError) as e:
                        self.logger.warning("Failed to create index for %s: %s", table_name, e)

            self.logger.info("All application tables created successfully")
            return True

        except (ImportError, AttributeError, ValueError) as e:
            self.logger.error("Failed to create application tables: %s", e)
            return False

    def insert(self, model: BaseModel) -> Optional[int]:
        """모델 인스턴스를 데이터베이스에 삽입"""
        try:
            db_type = self.config_db_manager.db_type
            query, values = model.get_insert_query(db_type)

            if db_type == "postgresql":
                query += " RETURNING id"
                result = self.config_db_manager.execute_query_one(query, tuple(values))
                return {"result": "success"}
            else:
                # SQLite의 경우 execute_insert 사용
                self.config_db_manager.execute_insert(query, tuple(values))
                return {"result": "success"}

        except AttributeError as e:
            self.logger.error("Failed to insert %s: %s", model.__class__.__name__, e)
            return None

    def update(self, model: BaseModel) -> bool:
        """모델 인스턴스를 데이터베이스에서 업데이트 (리스트 데이터 처리 지원)"""
        try:
            db_type = self.config_db_manager.db_type

            original_data = {}
            for attr_name in dir(model):
                if not attr_name.startswith('_') and not callable(getattr(model, attr_name)):
                    attr_value = getattr(model, attr_name)
                    if not attr_name in ['id', 'created_at', 'updated_at']:
                        original_data[attr_name] = attr_value

            query, values = model.get_update_query(db_type)
            set_part = query.split('SET')[1].split('WHERE')[0].strip()
            set_clauses = [clause.strip() for clause in set_part.split(',')]

            processed_values = []
            value_index = 0

            for clause in set_clauses:
                column_name = clause.split('=')[0].strip()
                original_value = original_data.get(column_name)

                if isinstance(original_value, list) and db_type == "postgresql":
                    escaped_items = []
                    for item in original_value:
                        escaped_item = str(item).replace('"', '""')
                        escaped_items.append(f'"{escaped_item}"')
                    array_literal = "{" + ",".join(escaped_items) + "}"
                    processed_values.append(array_literal)
                else:
                    processed_values.append(values[value_index])

                value_index += 1

            processed_values.append(values[-1])
            affected_rows = self.config_db_manager.execute_update_delete(query, tuple(processed_values))
            return {"result": "success"}

        except AttributeError as e:
            self.logger.error("Failed to update %s: %s", model.__class__.__name__, e)
            return False

    def delete(self, model_class: Type[BaseModel], record_id: int) -> bool:
        """ID로 레코드 삭제"""
        try:
            table_name = model_class().get_table_name()
            db_type = self.config_db_manager.db_type

            if db_type == "postgresql":
                query = f"DELETE FROM {table_name} WHERE id = %s"
            else:
                query = f"DELETE FROM {table_name} WHERE id = ?"

            affected_rows = self.config_db_manager.execute_update_delete(query, (record_id,))
            return affected_rows is not None and affected_rows > 0

        except AttributeError as e:
            self.logger.error("Failed to delete %s with id %s: %s",
                            model_class.__name__, record_id, e)
            return False

    def delete_by_condition(self, model_class: Type[BaseModel], conditions: Dict[str, Any]) -> bool:
        """조건으로 레코드 삭제"""
        try:
            table_name = model_class().get_table_name()
            db_type = self.config_db_manager.db_type

            # WHERE 조건 생성
            where_clauses = []
            values = []

            for key, value in conditions.items():
                if key.endswith("__like__"):
                    real_key = key.removesuffix("__like__")
                    if db_type == "postgresql":
                        where_clauses.append(f"{real_key} ILIKE %s")
                    else:
                        where_clauses.append(f"{real_key} LIKE ?")
                    values.append(f"%{value}%")
                elif key.endswith("__notlike__"):
                    real_key = key.removesuffix("__notlike__")
                    if db_type == "postgresql":
                        where_clauses.append(f"{real_key} NOT ILIKE %s")
                    else:
                        where_clauses.append(f"{real_key} NOT LIKE ?")
                    values.append(f"%{value}%")
                else:
                    if db_type == "postgresql":
                        where_clauses.append(f"{key} = %s")
                    else:
                        where_clauses.append(f"{key} = ?")
                    values.append(value)

            if not conditions:
                self.logger.warning("No conditions provided for delete_by_condition. Aborting to prevent full table deletion.")
                return False

            where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"

            query = f"DELETE FROM {table_name} WHERE {where_clause}"

            affected_rows = self.config_db_manager.execute_update_delete(query, tuple(values))
            return affected_rows is not None and affected_rows > 0

        except AttributeError as e:
            self.logger.error("Failed to delete %s by condition: %s", model_class.__name__, e)
            return False

    def find_by_id(self, model_class: Type[BaseModel], record_id: int, select_columns: List[str] = None, ignore_columns: List[str] = None) -> Optional[BaseModel]:
        """ID로 레코드 조회"""
        try:
            table_name = model_class().get_table_name()
            db_type = self.config_db_manager.db_type

            # SELECT 컬럼 설정
            if select_columns:
                columns_str = ", ".join(select_columns)
            elif ignore_columns:
                all_columns = ['id'] + list(model_class().get_schema().keys())
                filtered_columns = [col for col in all_columns if col not in ignore_columns]
                columns_str = ", ".join(filtered_columns) if filtered_columns else "*"
            else:
                columns_str = "*"

            if db_type == "postgresql":
                query = f"SELECT {columns_str} FROM {table_name} WHERE id = %s"
            else:
                query = f"SELECT {columns_str} FROM {table_name} WHERE id = ?"

            result = self.config_db_manager.execute_query_one(query, (record_id,))

            if result:
                return model_class.from_dict(dict(result))
            return None

        except AttributeError as e:
            self.logger.error("Failed to find %s with id %s: %s",
                            model_class.__name__, record_id, e)
            return None

    def find_all(self, model_class: Type[BaseModel], limit: int = 500, offset: int = 0, select_columns: List[str] = None, ignore_columns: List[str] = None) -> List[BaseModel]:
        """모든 레코드 조회 (페이징 지원)"""
        try:
            table_name = model_class().get_table_name()
            db_type = self.config_db_manager.db_type

            # SELECT 컬럼 설정
            if select_columns:
                columns_str = ", ".join(select_columns)
            elif ignore_columns:
                all_columns = ['id'] + list(model_class().get_schema().keys())
                filtered_columns = [col for col in all_columns if col not in ignore_columns]
                columns_str = ", ".join(filtered_columns) if filtered_columns else "*"
            else:
                columns_str = "*"

            if db_type == "postgresql":
                query = f"SELECT {columns_str} FROM {table_name} ORDER BY id DESC LIMIT %s OFFSET %s"
            else:
                query = f"SELECT {columns_str} FROM {table_name} ORDER BY id DESC LIMIT ? OFFSET ?"

            results = self.config_db_manager.execute_query(query, (limit, offset))

            return [model_class.from_dict(dict(row)) for row in results] if results else []

        except AttributeError as e:
            self.logger.error("Failed to find all %s: %s", model_class.__name__, e)
            return []

    def find_by_condition(self, model_class: Type[BaseModel],
                         conditions: Dict[str, Any],
                         limit: int = 500,
                         offset: int = 0,
                         orderby: str = "id",
                         orderby_asc: bool = False,
                         return_list: bool = False,
                         select_columns: List[str] = None,
                         ignore_columns: List[str] = None) -> List[BaseModel]:
        """조건으로 레코드 조회"""
        try:
            table_name = model_class().get_table_name()
            db_type = self.config_db_manager.db_type

            # WHERE 조건 생성
            where_clauses = []
            values = []

            for key, value in conditions.items():
                if key.endswith("__like__"):
                    real_key = key.removesuffix("__like__")
                    if db_type == "postgresql":
                        where_clauses.append(f"{real_key} ILIKE %s")
                    else:
                        where_clauses.append(f"{real_key} LIKE ?")
                    values.append(f"%{value}%")
                elif key.endswith("__notlike__"):
                    real_key = key.removesuffix("__notlike__")
                    if db_type == "postgresql":
                        where_clauses.append(f"{real_key} NOT ILIKE %s")
                    else:
                        where_clauses.append(f"{real_key} NOT LIKE ?")
                    values.append(f"%{value}%")
                else:
                    if db_type == "postgresql":
                        where_clauses.append(f"{key} = %s")
                    else:
                        where_clauses.append(f"{key} = ?")
                    values.append(value)

            where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"

            # LIMIT/OFFSET 추가
            if db_type == "postgresql":
                limit_clause = "LIMIT %s OFFSET %s"
            else:
                limit_clause = "LIMIT ? OFFSET ?"

            values.extend([limit, offset])
            orderby_type = "ASC" if orderby_asc else "DESC"

            # SELECT 컬럼 설정
            if select_columns:
                columns_str = ", ".join(select_columns)
            elif ignore_columns:
                all_columns = ['id'] + list(model_class().get_schema().keys())
                filtered_columns = [col for col in all_columns if col not in ignore_columns]
                columns_str = ", ".join(filtered_columns) if filtered_columns else "*"
            else:
                columns_str = "*"

            query = f"SELECT {columns_str} FROM {table_name} WHERE {where_clause} ORDER BY {orderby} {orderby_type} {limit_clause}"

            results = self.config_db_manager.execute_query(query, tuple(values))

            if return_list:
                return [row for row in results] if results else []
            else:
                return [model_class.from_dict(dict(row)) for row in results] if results else []

        except AttributeError as e:
            self.logger.error("Failed to find %s by condition: %s", model_class.__name__, e)
            return []

        except Exception as e:
            self.logger.error(f"Error finding by condition: {e}")
            return []


    def update_list_columns(self, model_class: Type[BaseModel], updates: Dict[str, Any], conditions: Dict[str, Any]) -> bool:
        """리스트 컬럼을 포함한 모델 업데이트

        Args:
            model: 업데이트할 모델 클래스 인스턴스
            updates: 업데이트할 컬럼과 값들의 딕셔너리
            conditions: WHERE 조건으로 사용할 컬럼과 값들의 딕셔너리

        Returns:
            bool: 성공 여부
        """
        try:
            table_name = model_class().get_table_name()
            db_type = self.config_db_manager.db_type

            # SET 절 생성
            set_clauses = []
            values = []

            for column, value in updates.items():
                # 리스트 데이터 처리
                if isinstance(value, list):
                    if db_type == "postgresql":
                        # PostgreSQL 배열 형식으로 변환: {item1,item2,item3}
                        escaped_items = []
                        for item in value:
                            # 특수문자 이스케이프 처리
                            escaped_item = str(item).replace('"', '""')
                            escaped_items.append(f'"{escaped_item}"')
                        array_literal = "{" + ",".join(escaped_items) + "}"
                        set_clauses.append(f"{column} = %s")
                        values.append(array_literal)
                    else:
                        # SQLite JSON 형식으로 변환
                        array_json = json.dumps(value)
                        set_clauses.append(f"{column} = ?")
                        values.append(array_json)
                else:
                    # 일반 데이터 처리
                    if db_type == "postgresql":
                        set_clauses.append(f"{column} = %s")
                    else:
                        set_clauses.append(f"{column} = ?")
                    values.append(value)

            # WHERE 절 생성
            where_clauses = []
            for key, value in conditions.items():
                if db_type == "postgresql":
                    where_clauses.append(f"{key} = %s")
                else:
                    where_clauses.append(f"{key} = ?")
                values.append(value)

            set_clause = ", ".join(set_clauses)
            where_clause = " AND ".join(where_clauses)

            query = f"UPDATE {table_name} SET {set_clause} WHERE {where_clause}"

            affected_rows = self.config_db_manager.execute_update_delete(query, tuple(values))
            return affected_rows is not None and affected_rows > 0

        except (AttributeError, ValueError) as e:
            self.logger.error("Failed to update list columns for %s: %s", model_class.__name__, e)
            return False

    def close(self):
        """데이터베이스 연결 종료"""
        if self.config_db_manager.connection:
            self.config_db_manager.connection.close()
            self.logger.info("Application database connection closed")

    def run_migrations(self) -> bool:
        """데이터베이스 스키마 마이그레이션 실행"""
        try:
            return self.config_db_manager.run_migrations(self._models_registry)
        except (AttributeError, ValueError) as e:
            self.logger.error("Failed to run migrations: %s", e)
            return False

    def get_table_list(self) -> List[Dict[str, Any]]:
        """데이터베이스의 모든 테이블 목록 조회"""
        try:
            db_type = self.config_db_manager.db_type

            if db_type == "postgresql":
                query = """
                    SELECT
                        schemaname as schema_name,
                        tablename as table_name,
                        tableowner as table_owner
                    FROM pg_tables
                    WHERE schemaname = 'public'
                    ORDER BY tablename
                """
                results = self.config_db_manager.execute_query(query)
            else:  # SQLite
                query = """
                    SELECT
                        name as table_name,
                        type as table_type,
                        sql as create_sql
                    FROM sqlite_master
                    WHERE type='table' AND name NOT LIKE 'sqlite_%'
                    ORDER BY name
                """
                results = self.config_db_manager.execute_query(query)

            return results if results else []

        except (AttributeError, ValueError) as e:
            self.logger.error("Failed to get table list: %s", e)
            return []

    def execute_raw_query(self, query: str, params: tuple = None) -> Dict[str, Any]:
        """임의의 SQL 쿼리 실행"""
        try:
            # 쿼리 정규화 (앞뒤 공백 제거, 세미콜론 제거)
            query = query.strip().rstrip(';')

            # 위험한 키워드 체크 (기본적인 보안)
            dangerous_keywords = [
                'DROP', 'DELETE', 'TRUNCATE', 'INSERT', 'UPDATE',
                'CREATE', 'ALTER', 'GRANT', 'REVOKE'
            ]

            query_upper = query.upper().strip()

            # SELECT 쿼리만 허용하는 기본 보안 체크
            if not query_upper.startswith('SELECT'):
                return {
                    "success": False,
                    "error": "Only SELECT queries are allowed",
                    "data": []
                }

            # 위험한 키워드가 포함되어 있는지 체크
            for keyword in dangerous_keywords:
                if keyword in query_upper:
                    return {
                        "success": False,
                        "error": f"Query contains forbidden keyword: {keyword}",
                        "data": []
                    }

            # 쿼리 길이 제한 (너무 긴 쿼리 방지)
            if len(query) > 1000:
                return {
                    "success": False,
                    "error": "Query too long (maximum 1000 characters)",
                    "data": []
                }

            results = self.config_db_manager.execute_query(query, params)

            if results is not None:
                if len(results) > 1000:
                    return {
                        "success": True,
                        "error": "Result truncated (showing first 1000 rows)",
                        "data": results[:1000],
                        "row_count": len(results),
                        "truncated": True
                    }

                return {
                    "success": True,
                    "error": None,
                    "data": results,
                    "row_count": len(results),
                    "truncated": False
                }
            else:
                return {
                    "success": False,
                    "error": "Query execution failed",
                    "data": []
                }

        except Exception as e:
            self.logger.error("Failed to execute raw query: %s", e)
            return {
                "success": False,
                "error": str(e),
                "data": []
            }

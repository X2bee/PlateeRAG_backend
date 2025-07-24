import logging
from typing import List, Dict, Any, Optional, Type
from config.database_manager import DatabaseManager
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
                return self.config_db_manager.execute_insert(query, tuple(values))

        except AttributeError as e:
            self.logger.error("Failed to insert %s: %s", model.__class__.__name__, e)
            return None

    def update(self, model: BaseModel) -> bool:
        """모델 인스턴스를 데이터베이스에서 업데이트"""
        try:
            db_type = self.config_db_manager.db_type
            query, values = model.get_update_query(db_type)
            affected_rows = self.config_db_manager.execute_update_delete(query, tuple(values))
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
                if db_type == "postgresql":
                    where_clauses.append(f"{key} = %s")
                else:
                    where_clauses.append(f"{key} = ?")
                values.append(value)

            where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"

            query = f"DELETE FROM {table_name} WHERE {where_clause}"

            affected_rows = self.config_db_manager.execute_update_delete(query, tuple(values))
            return affected_rows is not None and affected_rows > 0

        except AttributeError as e:
            self.logger.error("Failed to delete %s by condition: %s", model_class.__name__, e)
            return False

    def find_by_id(self, model_class: Type[BaseModel], record_id: int) -> Optional[BaseModel]:
        """ID로 레코드 조회"""
        try:
            table_name = model_class().get_table_name()
            db_type = self.config_db_manager.db_type

            if db_type == "postgresql":
                query = f"SELECT * FROM {table_name} WHERE id = %s"
            else:
                query = f"SELECT * FROM {table_name} WHERE id = ?"

            result = self.config_db_manager.execute_query_one(query, (record_id,))

            if result:
                return model_class.from_dict(dict(result))
            return None

        except AttributeError as e:
            self.logger.error("Failed to find %s with id %s: %s",
                            model_class.__name__, record_id, e)
            return None

    def find_all(self, model_class: Type[BaseModel], limit: int = 100, offset: int = 0) -> List[BaseModel]:
        """모든 레코드 조회 (페이징 지원)"""
        try:
            table_name = model_class().get_table_name()
            db_type = self.config_db_manager.db_type

            if db_type == "postgresql":
                query = f"SELECT * FROM {table_name} ORDER BY id DESC LIMIT $1 OFFSET $2"
            else:
                query = f"SELECT * FROM {table_name} ORDER BY id DESC LIMIT ? OFFSET ?"

            results = self.config_db_manager.execute_query(query, (limit, offset))

            return [model_class.from_dict(dict(row)) for row in results] if results else []

        except AttributeError as e:
            self.logger.error("Failed to find all %s: %s", model_class.__name__, e)
            return []

    def find_by_condition(self, model_class: Type[BaseModel],
                         conditions: Dict[str, Any],
                         limit: int = 100,
                         offset: int = 0,
                         orderby: str = "id",
                         orderby_asc: bool = False,
                         return_list: bool = False) -> List[BaseModel]:
        """조건으로 레코드 조회"""
        try:
            table_name = model_class().get_table_name()
            db_type = self.config_db_manager.db_type

            # WHERE 조건 생성
            where_clauses = []
            values = []

            for key, value in conditions.items():
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
            query = f"SELECT * FROM {table_name} WHERE {where_clause} ORDER BY {orderby} {orderby_type} {limit_clause}"

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


    def close(self):
        """데이터베이스 연결 종료"""
        if self.config_db_manager.connection:
            self.config_db_manager.connection.close()
            self.logger.info("Application database connection closed")

    def run_migrations(self) -> bool:
        """데이터베이스 스키마 마이그레이션 실행"""
        try:
            return self.config_db_manager.run_migrations(self._models_registry)
        except Exception as e:
            self.logger.error("Failed to run migrations: %s", e)
            return False

"""
기본 데이터 모델 클래스
"""
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, Optional, List
import json
import logging

logger = logging.getLogger("base-model")

class BaseModel(ABC):
    """모든 데이터 모델의 기본 클래스"""

    def __init__(self, **kwargs):
        self.id: Optional[int] = kwargs.get('id')
        self.created_at: Optional[datetime] = kwargs.get('created_at')
        self.updated_at: Optional[datetime] = kwargs.get('updated_at')

        # 추가 필드들을 동적으로 설정
        for key, value in kwargs.items():
            if not hasattr(self, key):
                setattr(self, key, value)

    @abstractmethod
    def get_table_name(self) -> str:
        """테이블 이름 반환"""
        pass

    @abstractmethod
    def get_schema(self) -> Dict[str, str]:
        """테이블 스키마 반환 (컬럼명: 타입)"""
        pass

    def to_dict(self) -> Dict[str, Any]:
        """객체를 딕셔너리로 변환"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, datetime):
                result[key] = value.isoformat()
            elif isinstance(value, (list, dict)):
                result[key] = json.dumps(value) if value else None
            else:
                result[key] = value
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """딕셔너리에서 객체 생성"""
        # datetime 필드 변환
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if 'updated_at' in data and isinstance(data['updated_at'], str):
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])

        return cls(**data)

    def get_insert_query(self, db_type: str = "sqlite") -> tuple:
        """INSERT 쿼리 생성"""
        data = self.to_dict()
        # id와 타임스탬프 제외 (자동 생성)
        data.pop('id', None)
        data.pop('created_at', None)
        data.pop('updated_at', None)

        columns = list(data.keys())
        values = list(data.values())

        if db_type == "postgresql":
            placeholders = ["%s" for _ in range(len(values))]
        else:  # sqlite
            placeholders = ["?" for _ in range(len(values))]

        query = f"""
        INSERT INTO {self.get_table_name()} ({', '.join(columns)})
        VALUES ({', '.join(placeholders)})
        """

        return query.strip(), values

    def get_update_query(self, db_type: str = "sqlite") -> tuple:
        """UPDATE 쿼리 생성"""
        if not self.id:
            raise ValueError("Cannot update record without ID")

        data = self.to_dict()
        # id와 created_at 제외
        data.pop('id', None)
        data.pop('created_at', None)
        data['updated_at'] = datetime.now().isoformat()

        columns = list(data.keys())
        values = list(data.values())

        if db_type == "postgresql":
            set_clauses = [f"{col} = %s" for col in columns]
            where_placeholder = "%s"
        else:  # sqlite
            set_clauses = [f"{col} = ?" for col in columns]
            where_placeholder = "?"

        query = f"""
        UPDATE {self.get_table_name()}
        SET {', '.join(set_clauses)}
        WHERE id = {where_placeholder}
        """

        values.append(self.id)
        return query.strip(), values

    @classmethod
    def get_create_table_query(cls, db_type: str = "sqlite") -> str:
        """CREATE TABLE 쿼리 생성"""
        instance = cls()
        schema = instance.get_schema()

        # 기본 컬럼들 추가
        base_columns = {
            'id': 'SERIAL PRIMARY KEY' if db_type == "postgresql" else 'INTEGER PRIMARY KEY AUTOINCREMENT',
            'created_at': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP',
            'updated_at': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP'
        }

        # PostgreSQL의 경우 updated_at 자동 업데이트 트리거 필요
        all_columns = {**base_columns, **schema}

        columns_def = []
        for col_name, col_type in all_columns.items():
            columns_def.append(f"{col_name} {col_type}")

        columns_str = ',\n            '.join(columns_def)
        query = f"""CREATE TABLE IF NOT EXISTS {instance.get_table_name()} (
            {columns_str}
        )"""

        return query.strip()

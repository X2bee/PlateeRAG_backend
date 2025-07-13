"""
Persistent Config 모델 - 설정 저장을 위한 데이터베이스 모델
"""
from typing import Dict, Any
from models.base_model import BaseModel

class PersistentConfigModel(BaseModel):
    """설정 데이터를 저장하기 위한 모델"""
    
    def __init__(self, config_path: str = "", config_value: str = "", 
                 data_type: str = "string", **kwargs):
        super().__init__(**kwargs)
        self.config_path = config_path
        self.config_value = config_value
        self.data_type = data_type
    
    def get_table_name(self) -> str:
        return "persistent_configs"
    
    def get_schema(self) -> Dict[str, str]:
        """테이블 스키마 반환"""
        return {
            'config_path': 'VARCHAR(255) UNIQUE NOT NULL',
            'config_value': 'TEXT',
            'data_type': 'VARCHAR(50) DEFAULT \'string\''
        }
    
    @classmethod
    def get_create_table_query(cls, db_type: str = "sqlite") -> str:
        """테이블 생성 SQL 쿼리 반환"""
        # 기본 테이블만 생성, 인덱스는 따로 처리
        return super().get_create_table_query(db_type)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PersistentConfigModel':
        """딕셔너리에서 모델 인스턴스 생성"""
        return cls(
            config_path=data.get('config_path', ''),
            config_value=data.get('config_value', ''),
            data_type=data.get('data_type', 'string'),
            **{k: v for k, v in data.items() if k not in ['config_path', 'config_value', 'data_type']}
        )
    
    def __str__(self):
        return f"PersistentConfigModel(path={self.config_path}, value={self.config_value})"
    
    def __repr__(self):
        return f"PersistentConfigModel(id={self.id}, config_path='{self.config_path}', config_value='{self.config_value}', data_type='{self.data_type}')"

"""
VectorDB 설정
"""
from typing import Dict
from config.base_config import BaseConfig, PersistentConfig, convert_to_bool, convert_to_int

class VectorDBConfig(BaseConfig):
    """VectorDB 설정 관리"""

    def initialize(self) -> Dict[str, PersistentConfig]:
        """VectorDB 관련 설정들을 초기화"""
        
        # Qdrant 호스트 (기본: localhost)
        self.QDRANT_HOST = self.create_persistent_config(
            env_name="QDRANT_HOST",
            config_path="vectordb.qdrant.host",
            default_value="localhost"
        )
        
        # Qdrant HTTP 포트 (기본: 6333)
        self.QDRANT_PORT = self.create_persistent_config(
            env_name="QDRANT_PORT",
            config_path="vectordb.qdrant.port",
            default_value=6333,
            type_converter=convert_to_int
        )
        
        # Qdrant gRPC 사용 여부 (기본: False)
        self.QDRANT_USE_GRPC = self.create_persistent_config(
            env_name="QDRANT_USE_GRPC",
            config_path="vectordb.qdrant.use_grpc",
            default_value=False,
            type_converter=convert_to_bool
        )
        
        # Qdrant gRPC 포트 (기본: 6334)
        self.QDRANT_GRPC_PORT = self.create_persistent_config(
            env_name="QDRANT_GRPC_PORT",
            config_path="vectordb.qdrant.grpc_port",
            default_value=6334,
            type_converter=convert_to_int
        )
        
        # Qdrant 인증용 API 키 (기본: 빈 문자열)
        self.QDRANT_API_KEY = self.create_persistent_config(
            env_name="QDRANT_API_KEY",
            config_path="vectordb.qdrant.api_key",
            default_value=""
        )
        
        # 컬렉션 이름 (기본: default_collection)
        self.COLLECTION_NAME = self.create_persistent_config(
            env_name="QDRANT_COLLECTION",
            config_path="vectordb.qdrant.collection",
            default_value="default_collection"
        )
        
        # 벡터 차원 (기본: 1536)
        self.VECTOR_DIMENSION = self.create_persistent_config(
            env_name="VECTOR_DIMENSION",
            config_path="vectordb.qdrant.vector_dimension",
            default_value=1536,
            type_converter=convert_to_int
        )
        
        # 복제본 수 (기본: 1)
        self.REPLICAS = self.create_persistent_config(
            env_name="QDRANT_REPLICAS",
            config_path="vectordb.qdrant.replicas",
            default_value=1,
            type_converter=convert_to_int
        )
        
        # 샤드 수 (기본: 1)
        self.SHARDS = self.create_persistent_config(
            env_name="QDRANT_SHARDS",
            config_path="vectordb.qdrant.shards",
            default_value=1,
            type_converter=convert_to_int
        )
        
        return self.configs

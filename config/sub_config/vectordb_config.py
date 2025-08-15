"""
VectorDB 설정
"""
from typing import Dict
import os
from config.base_config import BaseConfig, PersistentConfig, convert_to_bool, convert_to_int

class VectorDBConfig(BaseConfig):
    """VectorDB 설정 관리"""

    def __init__(self):
        super().__init__()
        self._openai_config = None  # OpenAI 설정 참조를 위한 속성

    def set_openai_config(self, openai_config):
        """OpenAI 설정 참조 설정"""
        self._openai_config = openai_config

    def initialize(self) -> Dict[str, PersistentConfig]:
        """VectorDB 관련 설정들을 초기화"""

        # Qdrant 호스트 (기본: localhost)
        self.QDRANT_HOST = self.create_persistent_config(
            env_name="QDRANT_HOST",
            config_path="vectordb.qdrant.host",
            default_value="host.docker.internal"
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

        # Qdrant 인증용 API 키 (기본: 빈 문자열 - 로컬에서는 불필요)
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

        # ===== 임베딩 설정 =====

        # 임베딩 제공자 (openai, huggingface, custom_http)
        self.EMBEDDING_PROVIDER = self.create_persistent_config(
            env_name="EMBEDDING_PROVIDER",
            config_path="vectordb.embedding.provider",
            default_value="huggingface"
        )

        # OpenAI 임베딩 모델 (OpenAI API 키는 openai_config에서 관리)
        self.OPENAI_EMBEDDING_MODEL = self.create_persistent_config(
            env_name="OPENAI_EMBEDDING_MODEL",
            config_path="vectordb.embedding.openai.model",
            default_value="text-embedding-3-small"
        )

        # HuggingFace 임베딩 설정
        self.HUGGINGFACE_MODEL_NAME = self.create_persistent_config(
            env_name="HUGGINGFACE_MODEL_NAME",
            config_path="vectordb.embedding.huggingface.model_name",
            default_value="Qwen/Qwen3-Embedding-0.6B"
        )

        self.HUGGINGFACE_EMBEDDING_MODEL_DEVICE = self.create_persistent_config(
            env_name="HUGGINGFACE_EMBEDDING_MODEL_DEVICE",
            config_path="vectordb.embedding.huggingface.model_device",
            default_value="cpu"
        )

        self.HUGGINGFACE_API_KEY = self.create_persistent_config(
            env_name="HUGGINGFACE_API_KEY",
            config_path="vectordb.embedding.huggingface_api_key",
            default_value=""
        )

        # Custom HTTP API 임베딩 설정 (vLLM 등)
        self.CUSTOM_EMBEDDING_URL = self.create_persistent_config(
            env_name="CUSTOM_EMBEDDING_URL",
            config_path="vectordb.embedding.custom.url",
            default_value="http://localhost:8000/v1"
        )

        self.CUSTOM_EMBEDDING_API_KEY = self.create_persistent_config(
            env_name="CUSTOM_EMBEDDING_API_KEY",
            config_path="vectordb.embedding.custom.api_key",
            default_value=""
        )

        self.CUSTOM_EMBEDDING_MODEL = self.create_persistent_config(
            env_name="CUSTOM_EMBEDDING_MODEL",
            config_path="vectordb.embedding.custom.model",
            default_value="text-embedding-ada-002"
        )

        # 임베딩 차원 자동 감지 여부
        self.AUTO_DETECT_EMBEDDING_DIM = self.create_persistent_config(
            env_name="AUTO_DETECT_EMBEDDING_DIM",
            config_path="vectordb.embedding.auto_detect_dimension",
            default_value=True,
            type_converter=convert_to_bool
        )

        # 설정 검증 및 초기화
        self._validate_and_fix_config()

        return self.configs

    def get_openai_api_key(self):
        """OpenAI API 키를 openai_config에서 가져오기"""
        if self._openai_config and hasattr(self._openai_config, 'API_KEY'):
            return self._openai_config.API_KEY.value
        # 환경변수에서 직접 확인 (fallback)
        return os.environ.get("OPENAI_API_KEY", "")

    def _validate_and_fix_config(self):
        """임베딩 설정 검증 및 자동 수정"""
        provider = self.EMBEDDING_PROVIDER.value.lower()

        # 지원되는 제공자 목록
        supported_providers = ["openai", "huggingface", "custom_http"]

        if provider not in supported_providers:
            # 잘못된 제공자인 경우 기본값으로 설정
            print(f"Warning: Unsupported embedding provider '{provider}'. Setting to 'huggingface'")
            self.EMBEDDING_PROVIDER.value = "huggingface"
            provider = "huggingface"

        # 제공자별 필수 설정 검증 및 대체
        if provider == "openai":
            if not self.get_openai_api_key():
                print("Warning: OpenAI API key not configured. Trying to find alternative...")
                provider = self._find_available_provider()
                self.EMBEDDING_PROVIDER.value = provider

        if provider == "custom_http":
            if not self.CUSTOM_EMBEDDING_URL.value or self.CUSTOM_EMBEDDING_URL.value == "http://localhost:8000/v1":
                print("Warning: Custom HTTP URL not properly configured. Trying to find alternative...")
                provider = self._find_available_provider()
                self.EMBEDDING_PROVIDER.value = provider

        # HuggingFace 설정 확인
        if provider == "huggingface":
            # HuggingFace 모델명 검증
            if not self.HUGGINGFACE_MODEL_NAME.value:
                self.HUGGINGFACE_MODEL_NAME.value = "sentence-transformers/all-MiniLM-L6-v2"

            # sentence-transformers 패키지 확인
            try:
                import sentence_transformers
                print(f"sentence-transformers available, using HuggingFace: {self.HUGGINGFACE_MODEL_NAME.value}")
            except ImportError:
                print("Warning: sentence-transformers not installed. Trying OpenAI as fallback...")
                if self.get_openai_api_key():
                    print("OpenAI API key found. Switching to OpenAI provider")
                    self.EMBEDDING_PROVIDER.value = "openai"
                    provider = "openai"
                else:
                    print("No OpenAI API key. Using HuggingFace anyway (will fail gracefully)")

        print(f"Final embedding provider: {self.EMBEDDING_PROVIDER.value}")

    def _find_available_provider(self) -> str:
        """사용 가능한 임베딩 제공자 찾기"""
        # OpenAI 체크
        if self.get_openai_api_key():
            print("Found OpenAI API key. Using OpenAI provider")
            return "openai"

        # sentence-transformers 체크
        try:
            import sentence_transformers
            print("Found sentence-transformers. Using HuggingFace provider")
            return "huggingface"
        except ImportError:
            pass

        # Custom HTTP 체크
        if self.CUSTOM_EMBEDDING_URL.value and self.CUSTOM_EMBEDDING_URL.value != "http://localhost:8000/v1":
            print("Found Custom HTTP URL. Using custom_http provider")
            return "custom_http"

        # 최후의 수단으로 HuggingFace (실패하더라도)
        print("No ideal provider found. Defaulting to HuggingFace")
        return "huggingface"

    def check_and_switch_to_best_provider(self):
        """현재 상황에서 최적의 제공자로 자동 전환"""
        current_provider = self.EMBEDDING_PROVIDER.value.lower()
        best_provider = self._find_available_provider()

        if current_provider != best_provider:
            old_provider = current_provider
            self.EMBEDDING_PROVIDER.value = best_provider
            self.EMBEDDING_PROVIDER.save()  # 데이터베이스에 저장
            print(f"Auto-switched embedding provider from {old_provider} to {best_provider}")
            return True
        return False

    def get_embedding_provider_status(self) -> dict:
        """현재 임베딩 제공자 상태 반환"""
        provider = self.EMBEDDING_PROVIDER.value.lower()

        status = {
            "current_provider": provider,
            "config_valid": True,
            "issues": []
        }

        if provider == "openai":
            if not self.get_openai_api_key():
                status["config_valid"] = False
                status["issues"].append("OpenAI API key not configured")

        elif provider == "custom_http":
            if not self.CUSTOM_EMBEDDING_URL.value:
                status["config_valid"] = False
                status["issues"].append("Custom HTTP URL not configured")

        return status

    def switch_embedding_provider(self, new_provider: str) -> bool:
        """임베딩 제공자 변경 (검증 포함)"""
        new_provider = new_provider.lower()
        supported_providers = ["openai", "huggingface", "custom_http"]

        if new_provider not in supported_providers:
            return False

        # 변경 전 검증
        if new_provider == "openai" and not self.get_openai_api_key():
            return False

        if new_provider == "custom_http" and not self.CUSTOM_EMBEDDING_URL.value:
            return False

        # 제공자 변경
        old_provider = self.EMBEDDING_PROVIDER.value
        self.EMBEDDING_PROVIDER.value = new_provider
        self.EMBEDDING_PROVIDER.save()  # 데이터베이스에 저장

        print(f"Embedding provider switched from {old_provider} to {new_provider}")
        return True

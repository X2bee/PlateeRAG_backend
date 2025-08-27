"""
VectorDB 설정
"""
from typing import Dict
import os
from config.base_config import BaseConfig, PersistentConfig, convert_to_bool, convert_to_int

class EmbeddingConfig(BaseConfig):
    """Embedding 설정 관리"""
    def initialize(self) -> Dict[str, PersistentConfig]:
        """Embedding 관련 설정들을 초기화"""
        self.AVAILABLE_EMBEDDING_LIST = self.create_persistent_config(
            env_name="AVAILABLE_EMBEDDING_LIST",
            config_path="embedding.available_embedding_list",
            default_value={}
        )
        self.EMBEDDING_PROVIDER = self.create_persistent_config(
            env_name="EMBEDDING_PROVIDER",
            config_path="embedding.provider",
            default_value="huggingface"
        )

        self.OPENAI_EMBEDDING_MODEL_NAME = self.create_persistent_config(
            env_name="OPENAI_EMBEDDING_MODEL_NAME",
            config_path="embedding.openai.model_name",
            default_value="text-embedding-3-small"
        )

        self.HUGGINGFACE_EMBEDDING_MODEL_NAME = self.create_persistent_config(
            env_name="HUGGINGFACE_EMBEDDING_MODEL_NAME",
            config_path="embedding.huggingface.model_name",
            default_value="Qwen/Qwen3-Embedding-0.6B"
        )

        self.HUGGINGFACE_EMBEDDING_MODEL_DEVICE = self.create_persistent_config(
            env_name="HUGGINGFACE_EMBEDDING_MODEL_DEVICE",
            config_path="embedding.huggingface.model_device",
            default_value="cpu"
        )

        self.CUSTOM_EMBEDDING_URL = self.create_persistent_config(
            env_name="CUSTOM_EMBEDDING_URL",
            config_path="embedding.custom.url",
            default_value="http://localhost:8000/v1"
        )

        self.CUSTOM_EMBEDDING_API_KEY = self.create_persistent_config(
            env_name="CUSTOM_EMBEDDING_API_KEY",
            config_path="embedding.custom.api_key",
            default_value=""
        )

        self.CUSTOM_EMBEDDING_MODEL_NAME = self.create_persistent_config(
            env_name="CUSTOM_EMBEDDING_MODEL_NAME",
            config_path="embedding.custom.model_name",
            default_value=""
        )

        self.AUTO_DETECT_EMBEDDING_DIM = self.create_persistent_config(
            env_name="AUTO_DETECT_EMBEDDING_DIM",
            config_path="vectordb.embedding.auto_detect_dimension",
            default_value=True,
            type_converter=convert_to_bool
        )

        return self.configs

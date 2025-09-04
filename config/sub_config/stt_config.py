"""
Speech-to-Text (STT) 설정 관리 모듈
"""
from typing import Dict
import os
from config.base_config import BaseConfig, PersistentConfig, convert_to_bool, convert_to_int

class STTConfig(BaseConfig):
    """STT 설정 관리"""
    def initialize(self) -> Dict[str, PersistentConfig]:
        self.AVAILABLE_STT_LIST = self.create_persistent_config(
            env_name="AVAILABLE_STT_LIST",
            config_path="stt.available_stt_list",
            default_value={}
        )
        self.STT_PROVIDER = self.create_persistent_config(
            env_name="STT_PROVIDER",
            config_path="stt.provider",
            default_value="huggingface"
        )
        self.OPENAI_STT_MODEL_NAME = self.create_persistent_config(
            env_name="OPENAI_STT_MODEL_NAME",
            config_path="stt.openai.model_name",
            default_value="whisper-1"
        )
        self.HUGGINGFACE_STT_MODEL_NAME = self.create_persistent_config(
            env_name="HUGGINGFACE_STT_MODEL_NAME",
            config_path="stt.huggingface.model_name",
            default_value="openai/whisper-small"
        )
        self.HUGGINGFACE_STT_MODEL_DEVICE = self.create_persistent_config(
            env_name="HUGGINGFACE_STT_MODEL_DEVICE",
            config_path="stt.huggingface.model_device",
            default_value="cpu"
        )

        return self.configs

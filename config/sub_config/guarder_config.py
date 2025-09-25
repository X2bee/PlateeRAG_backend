"""
Guarder 설정 관리
"""
from typing import Dict
from config.base_config import BaseConfig, PersistentConfig, convert_to_bool

class GuarderConfig(BaseConfig):
    """Guarder 설정 관리"""
    def initialize(self) -> Dict[str, PersistentConfig]:
        self.IS_AVAILABLE_GUARDER = self.create_persistent_config(
            env_name="IS_AVAILABLE_GUARDER",
            config_path="guarder.is_available_guarder",
            default_value=True,
            type_converter=convert_to_bool
        )
        self.GUARDER_PROVIDER = self.create_persistent_config(
            env_name="GUARDER_PROVIDER",
            config_path="guarder.provider",
            default_value="qwen3guard"
        )
        self.QWEN3GUARD_MODEL_NAME = self.create_persistent_config(
            env_name="QWEN3GUARD_MODEL_NAME",
            config_path="guarder.qwen3guard.model_name",
            default_value="Qwen/Qwen3Guard-Gen-0.6B"
        )
        self.QWEN3GUARD_MODEL_DEVICE = self.create_persistent_config(
            env_name="QWEN3GUARD_MODEL_DEVICE",
            config_path="guarder.qwen3guard.model_device",
            default_value="cpu"
        )

        return self.configs

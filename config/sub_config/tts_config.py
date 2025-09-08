"""
TTS 설정 관리
"""
from typing import Dict
from config.base_config import BaseConfig, PersistentConfig, convert_to_bool

class TTSConfig(BaseConfig):
    """TTS 설정 관리"""
    def initialize(self) -> Dict[str, PersistentConfig]:
        self.IS_AVAILABLE_TTS = self.create_persistent_config(
            env_name="IS_AVAILABLE_TTS",
            config_path="tts.is_available_tts",
            default_value=False,
            type_converter=convert_to_bool
        )
        self.AVAILABLE_TTS_LIST = self.create_persistent_config(
            env_name="AVAILABLE_TTS_LIST",
            config_path="tts.available_tts_list",
            default_value={}
        )
        self.TTS_PROVIDER = self.create_persistent_config(
            env_name="TTS_PROVIDER",
            config_path="tts.provider",
            default_value="zonos"
        )
        self.OPENAI_TTS_MODEL_NAME = self.create_persistent_config(
            env_name="OPENAI_TTS_MODEL_NAME",
            config_path="tts.openai.model_name",
            default_value="whisper-1"
        )
        self.ZONOS_TTS_MODEL_NAME = self.create_persistent_config(
            env_name="ZONOS_TTS_MODEL_NAME",
            config_path="tts.zonos.model_name",
            default_value="Zyphra/Zonos-v0.1-transformer"
        )
        self.ZONOS_TTS_MODEL_DEVICE = self.create_persistent_config(
            env_name="ZONOS_TTS_MODEL_DEVICE",
            config_path="tts.zonos.model_device",
            default_value="gpu"
        )
        self.ZONOS_TTS_DEFAULT_SPEAKER = self.create_persistent_config(
            env_name="ZONOS_TTS_DEFAULT_SPEAKER",
            config_path="tts.zonos.default_speaker",
            default_value="female_sample3"
        )

        return self.configs

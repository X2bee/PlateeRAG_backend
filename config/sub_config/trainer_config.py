"""
OpenAI API 관련 설정
"""
from typing import Dict
from config.base_config import BaseConfig, PersistentConfig

class TrainerConfig(BaseConfig):
    """Trainer 관련 설정 관리"""

    def initialize(self) -> Dict[str, PersistentConfig]:
        """Trainer 관련 설정들을 초기화"""

        self.TRAINER_HOST = self.create_persistent_config(
            env_name="TRAINER_HOST",
            config_path="trainer.host",
            default_value="polarag_trainer",
            file_path="polarag_trainer.txt"
        )

        self.TRAINER_PORT = self.create_persistent_config(
            env_name="TRAINER_PORT",
            config_path="trainer.port",
            default_value="8010",
            file_path="polarag_trainer.txt"
        )

        return self.configs

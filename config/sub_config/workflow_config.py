"""
워크플로우 관련 설정
"""
from typing import Dict
from config.base_config import BaseConfig, PersistentConfig, convert_to_bool, convert_to_int

class WorkflowConfig(BaseConfig):
    """워크플로우 실행 관련 설정 관리"""

    def initialize(self) -> Dict[str, PersistentConfig]:
        """워크플로우 관련 설정들을 초기화"""

        # 워크플로우 실행 타임아웃 (초)
        self.WORKFLOW_EXECUTION_TIMEOUT = self.create_persistent_config(
            env_name="WORKFLOW_EXECUTION_TIMEOUT",
            config_path="workflow.execution_timeout",
            default_value=300,  # 5분
            type_converter=convert_to_int
        )

        return self.configs

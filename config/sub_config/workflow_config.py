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
        self.EXECUTION_TIMEOUT = self.create_persistent_config(
            env_name="WORKFLOW_TIMEOUT",
            config_path="workflow.timeout",
            default_value=300,  # 5분
            type_converter=convert_to_int
        )
        
        # 최대 노드 개수
        self.MAX_NODES = self.create_persistent_config(
            env_name="MAX_WORKFLOW_NODES",
            config_path="workflow.max_nodes",
            default_value=1000,
            type_converter=convert_to_int
        )
        
        # 워크플로우 병렬 실행 허용 여부
        self.ALLOW_PARALLEL_EXECUTION = self.create_persistent_config(
            env_name="WORKFLOW_ALLOW_PARALLEL",
            config_path="workflow.allow_parallel",
            default_value=True,
            type_converter=convert_to_bool
        )
        
        # 워크플로우 결과 캐싱 여부
        self.ENABLE_RESULT_CACHING = self.create_persistent_config(
            env_name="WORKFLOW_ENABLE_CACHING",
            config_path="workflow.enable_caching",
            default_value=True,
            type_converter=convert_to_bool
        )
        
        # 최대 동시 실행 워크플로우 수
        self.MAX_CONCURRENT_WORKFLOWS = self.create_persistent_config(
            env_name="MAX_CONCURRENT_WORKFLOWS",
            config_path="workflow.max_concurrent",
            default_value=5,
            type_converter=convert_to_int
        )
        
        # 워크플로우 로그 저장 여부
        self.SAVE_EXECUTION_LOGS = self.create_persistent_config(
            env_name="WORKFLOW_SAVE_LOGS",
            config_path="workflow.save_logs",
            default_value=True,
            type_converter=convert_to_bool
        )
        
        return self.configs

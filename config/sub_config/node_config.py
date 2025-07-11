"""
노드 관련 설정
"""
from typing import Dict
from config.base_config import BaseConfig, PersistentConfig, convert_to_bool, convert_to_int

class NodeConfig(BaseConfig):
    """노드 시스템 관련 설정 관리"""
    
    def initialize(self) -> Dict[str, PersistentConfig]:
        """노드 관련 설정들을 초기화"""
        
        # 노드 캐싱 활성화 여부
        self.CACHE_ENABLED = self.create_persistent_config(
            env_name="NODE_CACHE_ENABLED",
            config_path="node.cache_enabled",
            default_value=True,
            type_converter=convert_to_bool
        )
        
        # 노드 discovery 자동 실행 여부
        self.AUTO_DISCOVERY = self.create_persistent_config(
            env_name="NODE_AUTO_DISCOVERY",
            config_path="node.auto_discovery",
            default_value=True,
            type_converter=convert_to_bool
        )
        
        # 노드 검증 활성화 여부
        self.VALIDATION_ENABLED = self.create_persistent_config(
            env_name="NODE_VALIDATION_ENABLED",
            config_path="node.validation_enabled",
            default_value=True,
            type_converter=convert_to_bool
        )
        
        # 노드 실행 타임아웃 (초)
        self.EXECUTION_TIMEOUT = self.create_persistent_config(
            env_name="NODE_EXECUTION_TIMEOUT",
            config_path="node.execution_timeout",
            default_value=60,
            type_converter=convert_to_int
        )
        
        # 노드 레지스트리 파일 경로
        self.REGISTRY_FILE_PATH = self.create_persistent_config(
            env_name="NODE_REGISTRY_FILE_PATH",
            config_path="node.registry_file_path",
            default_value="constants/exported_nodes.json"
        )
        
        # 노드 디버그 모드
        self.DEBUG_MODE = self.create_persistent_config(
            env_name="NODE_DEBUG_MODE",
            config_path="node.debug_mode",
            default_value=False,
            type_converter=convert_to_bool
        )
        
        return self.configs

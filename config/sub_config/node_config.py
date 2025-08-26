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

        # 노드 레지스트리 파일 경로
        self.REGISTRY_FILE_PATH = self.create_persistent_config(
            env_name="NODE_REGISTRY_FILE_PATH",
            config_path="node.registry_file_path",
            default_value="constants/exported_nodes.json"
        )


        return self.configs

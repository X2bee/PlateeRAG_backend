"""
노드 관련 설정
"""
from typing import Dict
from config.base_config import BaseConfig, PersistentConfig, convert_to_bool, convert_to_int

class NodeConfig(BaseConfig):
    """노드 시스템 관련 설정 관리"""

    def initialize(self) -> Dict[str, PersistentConfig]:
        """노드 관련 설정들을 초기화"""

        # 노드 레지스트리 파일 경로
        self.REGISTRY_FILE_PATH = self.create_persistent_config(
            env_name="NODE_REGISTRY_FILE_PATH",
            config_path="node.registry_file_path",
            default_value="constants/exported_nodes.json"
        )


        return self.configs

"""
Config Composer - 모든 설정을 통합 관리
"""
import logging
from typing import Dict, Any
from config.sub_config.openai_config import OpenAIConfig
from config.sub_config.app_config import AppConfig
from config.sub_config.workflow_config import WorkflowConfig
from config.sub_config.node_config import NodeConfig
from config.sub_config.database_config import DatabaseConfig
from config.persistent_config import PersistentConfig, export_config_summary, refresh_all_configs, save_all_configs
from config.database_manager import initialize_database

logger = logging.getLogger("config-composer")

class ConfigComposer:
    """
    모든 설정을 통합적으로 관리하는 클래스
    """
    
    def __init__(self):
        self.openai: OpenAIConfig = OpenAIConfig()
        self.app: AppConfig = AppConfig()
        self.workflow: WorkflowConfig = WorkflowConfig()
        self.node: NodeConfig = NodeConfig()
        self.database: DatabaseConfig = DatabaseConfig()
        
        # 모든 설정을 저장하는 딕셔너리
        self.all_configs: Dict[str, PersistentConfig] = {}
        
        self.logger = logger
    
    def initialize_all_configs(self) -> Dict[str, Any]:
        """
        모든 설정을 초기화하고 app.state에 저장할 데이터 구조 반환
        """
        try:
            self.logger.info("Initializing all configurations...")
            
            # 1. 먼저 데이터베이스 설정 초기화
            database_configs = self.database.initialize()
            self.all_configs.update(database_configs)
            
            # 2. 데이터베이스 초기화 (마이그레이션 포함) - 다른 설정들이 DB를 사용할 수 있도록
            db_initialized = initialize_database(self.database)
            if db_initialized:
                self.logger.info("Database initialized successfully")
            else:
                self.logger.warning("Database initialization failed, using JSON fallback")
            
            # 3. 이제 나머지 설정 카테고리 초기화 (DB 연결 상태에서)
            openai_configs = self.openai.initialize()
            app_configs = self.app.initialize()
            workflow_configs = self.workflow.initialize()
            node_configs = self.node.initialize()
            
            # 모든 설정을 하나의 딕셔너리로 통합
            self.all_configs.update(openai_configs)
            self.all_configs.update(app_configs)
            self.all_configs.update(workflow_configs)
            self.all_configs.update(node_configs)
            
            self.logger.info(f"Successfully initialized {len(self.all_configs)} configurations")
            
            # app.state에 저장할 구조화된 데이터 반환
            return {
                "openai": self.openai,
                "app": self.app,
                "workflow": self.workflow,
                "node": self.node,
                "database": self.database,
                "all_configs": self.all_configs
            }
            
        except Exception as e:
            self.logger.error(f"Failed to initialize configurations: {e}")
            raise
    
    def get_config_by_name(self, config_name: str) -> PersistentConfig:
        """
        이름으로 특정 설정 가져오기
        """
        if config_name in self.all_configs:
            return self.all_configs[config_name]
        raise KeyError(f"Configuration '{config_name}' not found")
    
    def get_config_summary(self) -> Dict[str, Any]:
        """
        모든 설정의 요약 정보 반환
        """
        return {
            "total_configs": len(self.all_configs),
            "categories": {
                "openai": self.openai.get_config_summary(),
                "app": self.app.get_config_summary(),
                "workflow": self.workflow.get_config_summary(),
                "node": self.node.get_config_summary(),
                "database": self.database.get_config_summary()
            },
            "persistent_summary": export_config_summary()
        }
    
    def refresh_all(self):
        """
        모든 설정을 데이터베이스에서 다시 로드
        """
        self.logger.info("Refreshing all configurations from database...")
        refresh_all_configs()
        self.logger.info("All configurations refreshed successfully")
    
    def save_all(self):
        """
        모든 설정을 데이터베이스에 저장
        """
        self.logger.info("Saving all configurations to database...")
        save_all_configs()
        self.logger.info("All configurations saved successfully")
    
    def ensure_directories(self):
        """
        필요한 디렉토리들 생성
        """
        import os
        directories = self.app.DATA_DIRECTORIES.value
        
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                self.logger.info(f"Created directory: {directory}")
            else:
                self.logger.info(f"Directory already exists: {directory}")
    
    def validate_critical_configs(self) -> Dict[str, Any]:
        """
        중요한 설정들이 올바르게 설정되었는지 검증
        """
        validation_results = {
            "valid": True,
            "warnings": [],
            "errors": []
        }
        
        # OpenAI API 키 검증
        if not self.openai.API_KEY.value or not self.openai.API_KEY.value.strip():
            validation_results["warnings"].append("OpenAI API key is not configured")
        
        # 포트 범위 검증
        port = self.app.PORT.value
        if not (1 <= port <= 65535):
            validation_results["errors"].append(f"Invalid port number: {port}")
            validation_results["valid"] = False
        
        # 타임아웃 값 검증
        workflow_timeout = self.workflow.EXECUTION_TIMEOUT.value
        if workflow_timeout <= 0:
            validation_results["errors"].append(f"Invalid workflow timeout: {workflow_timeout}")
            validation_results["valid"] = False
        
        return validation_results

# 전역 설정 컴포저 인스턴스
config_composer = ConfigComposer()

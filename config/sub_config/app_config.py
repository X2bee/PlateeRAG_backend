"""
애플리케이션 기본 설정
"""
from typing import Dict
from config.base_config import BaseConfig, PersistentConfig, convert_to_bool

class AppConfig(BaseConfig):
    """애플리케이션 기본 설정 관리"""
    
    def initialize(self) -> Dict[str, PersistentConfig]:
        """애플리케이션 기본 설정들을 초기화"""
        
        # 애플리케이션 환경
        self.ENVIRONMENT = self.create_persistent_config(
            env_name="APP_ENVIRONMENT",
            config_path="app.environment",
            default_value="development"
        )
        
        # 디버그 모드
        self.DEBUG_MODE = self.create_persistent_config(
            env_name="DEBUG_MODE",
            config_path="app.debug_mode",
            default_value=False,
            type_converter=convert_to_bool
        )
        
        # 로그 레벨
        self.LOG_LEVEL = self.create_persistent_config(
            env_name="LOG_LEVEL",
            config_path="app.log_level",
            default_value="INFO"
        )
        
        # 서버 호스트
        self.HOST = self.create_persistent_config(
            env_name="APP_HOST",
            config_path="app.host",
            default_value="0.0.0.0"
        )
        
        # 서버 포트
        self.PORT = self.create_persistent_config(
            env_name="APP_PORT",
            config_path="app.port",
            default_value=8000,
            type_converter=int
        )
        
        # CORS 허용 오리진
        self.CORS_ORIGINS = self.create_persistent_config(
            env_name="CORS_ORIGINS",
            config_path="app.cors_origins",
            default_value=["*"]
        )
        
        # 데이터 디렉토리들
        self.DATA_DIRECTORIES = self.create_persistent_config(
            env_name="DATA_DIRECTORIES",
            config_path="app.data_directories",
            default_value=["constants", "downloads"]
        )
        
        # 로그 포맷
        self.LOG_FORMAT = self.create_persistent_config(
            env_name="LOG_FORMAT",
            config_path="app.log_format",
            default_value="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        
        return self.configs

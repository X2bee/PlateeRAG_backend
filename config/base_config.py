"""
Base configuration classes and utilities
"""
import os
import logging
from typing import Any, Optional, Union, List, Dict
from abc import ABC, abstractmethod
from config.persistent_config import PersistentConfig

logger = logging.getLogger("config-base")

class BaseConfig(ABC):
    """
    모든 설정 클래스의 기본 클래스
    """
    
    def __init__(self):
        self.configs: Dict[str, PersistentConfig] = {}
        self.logger = logging.getLogger(f"config-{self.__class__.__name__.lower()}")
        
        # 설정 자동 초기화
        try:
            self.initialize()
        except Exception as e:
            self.logger.error(f"Failed to initialize config: {e}")
            raise
    
    @abstractmethod
    def initialize(self) -> Dict[str, PersistentConfig]:
        """
        설정을 초기화하고 PersistentConfig 객체들을 반환
        """
        pass
    
    def get_env_value(self, env_name: str, default_value: Any, 
                     file_path: Optional[str] = None, 
                     type_converter: Optional[callable] = None) -> Any:
        """
        환경변수에서 값을 가져오고, 없으면 파일에서 읽거나 기본값 사용
        
        Args:
            env_name: 환경변수 이름
            default_value: 기본값
            file_path: 파일 경로 (선택사항)
            type_converter: 타입 변환 함수 (선택사항)
        
        Returns:
            설정값
        """
        # 1. 환경변수에서 확인
        env_value = os.environ.get(env_name)
        if env_value is not None:
            try:
                if type_converter:
                    converted_value = type_converter(env_value)
                    self.logger.info(f"'{env_name}' loaded from environment: {converted_value}")
                    return converted_value
                else:
                    self.logger.info(f"'{env_name}' loaded from environment: {env_value}")
                    return env_value
            except (ValueError, TypeError) as e:
                self.logger.warning(f"Failed to convert environment value for '{env_name}': {e}")
        
        # 2. 파일에서 확인 (제공된 경우)
        if file_path and os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_value = f.read().strip()
                    if file_value:
                        if type_converter:
                            converted_value = type_converter(file_value)
                            self.logger.info(f"'{env_name}' loaded from file {file_path}: {converted_value}")
                            return converted_value
                        else:
                            os.environ[env_name] = file_value  # 환경변수에도 설정
                            self.logger.info(f"'{env_name}' loaded from file {file_path}: {file_value}")
                            return file_value
            except Exception as e:
                self.logger.warning(f"Failed to read {file_path} for '{env_name}': {e}")
        
        # 3. 기본값 사용
        self.logger.info(f"'{env_name}' using default value: {default_value}")
        return default_value
    
    def create_persistent_config(self, env_name: str, config_path: str, 
                               default_value: Any, file_path: Optional[str] = None,
                               type_converter: Optional[callable] = None) -> PersistentConfig:
        """
        PersistentConfig 객체 생성
        """
        env_value = self.get_env_value(env_name, default_value, file_path, type_converter)
        config = PersistentConfig(
            env_name=env_name,
            config_path=config_path,
            env_value=env_value
        )
        self.configs[env_name] = config
        return config
    
    def __getitem__(self, key: str) -> PersistentConfig:
        """
        설정에 딕셔너리 형태로 접근할 수 있도록 지원
        """
        if key in self.configs:
            return self.configs[key]
        raise KeyError(f"Configuration '{key}' not found in {self.__class__.__name__}")

    def get_config_summary(self) -> Dict[str, Any]:
        """
        이 설정 클래스의 요약 정보 반환
        """
        return {
            "class_name": self.__class__.__name__,
            "config_count": len(self.configs),
            "configs": {
                name: {
                    "current_value": config.value,
                    "default_value": config.env_value,
                    "config_path": config.config_path
                }
                for name, config in self.configs.items()
            }
        }

def convert_to_bool(value: str) -> bool:
    """문자열을 bool로 변환"""
    return value.lower() in ('true', '1', 'yes', 'on', 'enabled')

def convert_to_int(value: str) -> int:
    """문자열을 int로 변환"""
    return int(value)

def convert_to_float(value: str) -> float:
    """문자열을 float로 변환"""
    return float(value)

def convert_to_list(value: str, separator: str = ',') -> List[str]:
    """문자열을 리스트로 변환"""
    return [item.strip() for item in value.split(separator) if item.strip()]

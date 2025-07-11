import json
import os
import logging
from typing import Generic, TypeVar, Any, List, Dict, Optional
from pathlib import Path

T = TypeVar('T')

logger = logging.getLogger("persistent-config")

# 전역 설정 레지스트리
PERSISTENT_CONFIG_REGISTRY: List['PersistentConfig'] = []

# 설정 데이터를 저장할 파일 경로
CONFIG_DB_PATH = "constants/config.json"

def ensure_config_directory():
    """설정 파일 디렉토리가 존재하는지 확인하고 없으면 생성"""
    config_dir = os.path.dirname(CONFIG_DB_PATH)
    if config_dir and not os.path.exists(config_dir):
        os.makedirs(config_dir, exist_ok=True)
        logger.info(f"Created config directory: {config_dir}")

def load_config_data() -> Dict[str, Any]:
    """JSON 파일에서 설정 데이터를 로드"""
    ensure_config_directory()
    
    if not os.path.exists(CONFIG_DB_PATH):
        logger.info(f"Config file not found at {CONFIG_DB_PATH}, creating new one")
        return {}
    
    try:
        with open(CONFIG_DB_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
            logger.info(f"Loaded config data from {CONFIG_DB_PATH}")
            return data
    except (json.JSONDecodeError, FileNotFoundError) as e:
        logger.warning(f"Failed to load config data: {e}, using empty config")
        return {}

def save_config_data(config_data: Dict[str, Any]):
    """설정 데이터를 JSON 파일에 저장"""
    ensure_config_directory()
    
    try:
        with open(CONFIG_DB_PATH, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved config data to {CONFIG_DB_PATH}")
    except Exception as e:
        logger.error(f"Failed to save config data: {e}")
        raise

def get_config_value(config_path: str) -> Optional[Any]:
    """점으로 구분된 경로를 사용하여 설정 값을 가져옴"""
    config_data = load_config_data()
    
    path_parts = config_path.split(".")
    current_data = config_data
    
    try:
        for key in path_parts:
            current_data = current_data[key]
        return current_data
    except (KeyError, TypeError):
        return None

def set_config_value(config_path: str, value: Any):
    """점으로 구분된 경로를 사용하여 설정 값을 저장"""
    config_data = load_config_data()
    
    path_parts = config_path.split(".")
    current_data = config_data
    
    # 중간 경로들 생성
    for key in path_parts[:-1]:
        if key not in current_data:
            current_data[key] = {}
        current_data = current_data[key]
    
    # 최종 값 설정
    current_data[path_parts[-1]] = value
    
    # 파일에 저장
    save_config_data(config_data)

class PersistentConfig(Generic[T]):
    """
    데이터베이스와 연동되는 지속적 설정 클래스
    환경 변수와 저장된 설정을 관리합니다.
    """
    
    def __init__(self, env_name: str, config_path: str, env_value: T):
        self.env_name = env_name
        self.config_path = config_path
        self.env_value = env_value
        
        # 저장된 설정 값 확인
        self.config_value = get_config_value(config_path)
        
        if self.config_value is not None:
            logger.info(f"'{env_name}' loaded from database: {self.config_value}")
            self.value = self.config_value
        else:
            logger.info(f"'{env_name}' using default value: {env_value}")
            self.value = env_value
        
        # 전역 레지스트리에 등록
        PERSISTENT_CONFIG_REGISTRY.append(self)
    
    def __str__(self):
        return str(self.value)
    
    def __repr__(self):
        return f"PersistentConfig(env_name='{self.env_name}', value={self.value})"
    
    @property
    def __dict__(self):
        raise TypeError(
            "PersistentConfig object cannot be converted to dict, use config_get or .value instead."
        )
    
    def __getattribute__(self, item):
        if item == "__dict__":
            raise TypeError(
                "PersistentConfig object cannot be converted to dict, use config_get or .value instead."
            )
        return super().__getattribute__(item)
    
    def update(self):
        """데이터베이스에서 최신 값을 다시 로드"""
        new_value = get_config_value(self.config_path)
        if new_value is not None:
            self.value = new_value
            self.config_value = new_value
            logger.info(f"Updated {self.env_name} to new value: {self.value}")
        else:
            logger.warning(f"No saved value found for {self.env_name}, keeping current value")
    
    def save(self):
        """현재 값을 데이터베이스에 저장"""
        logger.info(f"Saving '{self.env_name}' to database: {self.value}")
        set_config_value(self.config_path, self.value)
        self.config_value = self.value
    
    def reset_to_default(self):
        """기본값으로 재설정"""
        logger.info(f"Resetting '{self.env_name}' to default value: {self.env_value}")
        self.value = self.env_value
        self.save()

# 유틸리티 함수들
def get_all_persistent_configs() -> List[PersistentConfig]:
    """등록된 모든 PersistentConfig 객체 반환"""
    return PERSISTENT_CONFIG_REGISTRY.copy()

def refresh_all_configs():
    """모든 PersistentConfig 객체를 데이터베이스에서 다시 로드"""
    logger.info("Refreshing all persistent configs...")
    for config in PERSISTENT_CONFIG_REGISTRY:
        config.update()

def save_all_configs():
    """모든 PersistentConfig 객체를 데이터베이스에 저장"""
    logger.info("Saving all persistent configs...")
    for config in PERSISTENT_CONFIG_REGISTRY:
        config.save()

def export_config_summary() -> Dict[str, Any]:
    """모든 설정의 요약 정보 반환"""
    return {
        "total_configs": len(PERSISTENT_CONFIG_REGISTRY),
        "config_file": CONFIG_DB_PATH,
        "configs": [
            {
                "env_name": config.env_name,
                "config_path": config.config_path,
                "current_value": config.value,
                "default_value": config.env_value,
                "is_saved": config.config_value is not None
            }
            for config in PERSISTENT_CONFIG_REGISTRY
        ]
    }

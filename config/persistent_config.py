import json
import os
import logging
from typing import Generic, TypeVar, Any, List, Dict, Optional

T = TypeVar('T')

logger = logging.getLogger("persistent-config")

# 전역 설정 레지스트리
PERSISTENT_CONFIG_REGISTRY: List['PersistentConfig'] = []
# 설정 데이터를 저장할 파일 경로 (데이터베이스 실패 시 fallback)
CONFIG_DB_PATH = "constants/config.json"
# DISABLE_JSON_FALLBACK=true 로 설정하면 JSON fallback을 비활성화
JSON_FALLBACK_ENABLED = os.environ.get("DISABLE_JSON_FALLBACK", "false").lower() != "true"
if JSON_FALLBACK_ENABLED:
    logger.info("JSON fallback is ENABLED - will use constants/config.json if database connection fails")
else:
    logger.warning("JSON fallback is DISABLED - application will fail if database connection is unavailable")

def ensure_config_directory():
    """설정 파일 디렉토리가 존재하는지 확인하고 없으면 생성"""
    config_dir = os.path.dirname(CONFIG_DB_PATH)
    if config_dir and not os.path.exists(config_dir):
        os.makedirs(config_dir, exist_ok=True)
        logger.info("Created config directory: %s", config_dir)

def load_config_data() -> Dict[str, Any]:
    """JSON 파일에서 설정 데이터를 로드 (fallback)"""
    ensure_config_directory()

    if not os.path.exists(CONFIG_DB_PATH):
        logger.info("Config file not found at %s, creating new one", CONFIG_DB_PATH)
        return {}

    try:
        with open(CONFIG_DB_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
            logger.info("Loaded config data from %s", CONFIG_DB_PATH)
            return data
    except (json.JSONDecodeError, FileNotFoundError) as e:
        logger.warning("Failed to load config data: %s, using empty config", e)
        return {}

def save_config_data(config_data: Dict[str, Any]):
    """설정 데이터를 JSON 파일에 저장 (fallback)"""
    ensure_config_directory()

    try:
        with open(CONFIG_DB_PATH, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        logger.info("Saved config data to %s", CONFIG_DB_PATH)
    except Exception as e:
        logger.error("Failed to save config data: %s", e)
        raise

def get_config_value_from_db(config_path: str) -> Optional[Any]:
    """데이터베이스에서 설정 값을 가져옴"""
    try:
        from config.database_manager import get_database_manager

        db_manager = get_database_manager()
        if not db_manager.connection:
            # 데이터베이스 연결이 없을 때 JSON fallback 사용 여부 확인
            if not JSON_FALLBACK_ENABLED:
                logger.error("Database connection failed and JSON fallback is disabled (DISABLE_JSON_FALLBACK=true)")
                raise ConnectionError("Database connection failed and JSON fallback is disabled")

            logger.info("Database connection unavailable, using JSON fallback for config: %s", config_path)
            return get_config_value_from_json(config_path)

        logger.debug("Fetching config from database: %s", config_path)

        if db_manager.db_type == "postgresql":
            query = "SELECT config_value, data_type FROM persistent_configs WHERE config_path = %s"
        else:  # sqlite
            query = "SELECT config_value, data_type FROM persistent_configs WHERE config_path = ?"

        result = db_manager.execute_query(query, (config_path,))

        if result and len(result) > 0:
            config_value = result[0]['config_value']
            data_type = result[0].get('data_type', 'string')

            logger.debug("Config loaded from database: %s = %s (type: %s)", config_path, config_value, data_type)

            # 타입에 따라 값 변환
            if data_type == 'json':
                return json.loads(config_value) if config_value else None
            elif data_type == 'boolean':
                return config_value.lower() in ('true', '1', 'yes') if isinstance(config_value, str) else bool(config_value)
            elif data_type == 'integer':
                return int(config_value)
            elif data_type == 'float':
                return float(config_value)
            else:
                return config_value

        logger.debug("Config not found in database: %s", config_path)
        return None

    except Exception as e:
        if not JSON_FALLBACK_ENABLED:
            logger.error("Failed to get config from database and JSON fallback is disabled: %s", e)
            raise ConnectionError(f"Failed to get config from database and JSON fallback is disabled: {e}")

        logger.warning("Failed to get config '%s' from database: %s, using JSON fallback", config_path, e)
        return get_config_value_from_json(config_path)

def set_config_value_to_db(config_path: str, value: Any):
    """데이터베이스에 설정 값을 저장"""
    try:
        from config.database_manager import get_database_manager

        db_manager = get_database_manager()
        if not db_manager.connection:
            # 데이터베이스 연결이 없을 때 JSON fallback 사용 여부 확인
            if not JSON_FALLBACK_ENABLED:
                logger.error("Database connection failed and JSON fallback is disabled (DISABLE_JSON_FALLBACK=true)")
                raise ConnectionError("Database connection failed and JSON fallback is disabled")

            logger.info("Database connection unavailable, using JSON fallback to save config: %s", config_path)
            return set_config_value_to_json(config_path, value)

        # 데이터 타입 결정
        if isinstance(value, bool):
            data_type = 'boolean'
            config_value = str(value)
        elif isinstance(value, int):
            data_type = 'integer'
            config_value = str(value)
        elif isinstance(value, float):
            data_type = 'float'
            config_value = str(value)
        elif isinstance(value, (list, dict)):
            data_type = 'json'
            config_value = json.dumps(value)
        else:
            data_type = 'string'
            config_value = str(value)

        logger.debug("Saving config to database: %s = %s (type: %s)", config_path, value, data_type)

        if db_manager.db_type == "postgresql":
            query = """
                INSERT INTO persistent_configs (config_path, config_value, data_type, updated_at)
                VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT (config_path)
                DO UPDATE SET config_value = EXCLUDED.config_value,
                             data_type = EXCLUDED.data_type,
                             updated_at = CURRENT_TIMESTAMP
            """
        else:  # sqlite
            query = """
                INSERT OR REPLACE INTO persistent_configs (config_path, config_value, data_type, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            """

        db_manager.execute_query(query, (config_path, config_value, data_type))
        logger.info("Config successfully saved to database: %s = %s", config_path, value)

    except Exception as e:
        if not JSON_FALLBACK_ENABLED:
            logger.error("Failed to save config to database and JSON fallback is disabled: %s", e)
            raise ConnectionError(f"Failed to save config to database and JSON fallback is disabled: {e}")

        logger.warning("Failed to save config '%s' to database: %s, using JSON fallback", config_path, e)
        set_config_value_to_json(config_path, value)

def get_config_value_from_json(config_path: str) -> Optional[Any]:
    """JSON 파일에서 설정 값을 가져옴 (fallback)"""
    config_data = load_config_data()

    path_parts = config_path.split(".")
    current_data = config_data

    try:
        for key in path_parts:
            current_data = current_data[key]
        return current_data
    except (KeyError, TypeError):
        return None

def set_config_value_to_json(config_path: str, value: Any):
    """JSON 파일에 설정 값을 저장 (fallback)"""
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

# 호환성을 위한 별칭
get_config_value = get_config_value_from_db
set_config_value = set_config_value_to_db

class PersistentConfig(Generic[T]):
    """
    데이터베이스와 연동되는 지속적 설정 클래스
    환경 변수와 저장된 설정을 관리합니다.
    """

    def __init__(self, env_name: str, config_path: str, env_value: T, type_converter: Optional[callable] = None):
        self.env_name = env_name
        self.config_path = config_path
        self.env_value = env_value
        self.type_converter = type_converter
        self.config_value = get_config_value(config_path)

        if self.config_value is not None:
            logger.info("'%s' loaded from database: %s", env_name, self.config_value)
            # type_converter가 있고 config_value가 문자열인 경우 변환 적용
            if self.type_converter and isinstance(self.config_value, str):
                try:
                    self.value = self.type_converter(self.config_value)
                    logger.debug("Applied type converter to loaded value for '%s': %s -> %s",
                               env_name, self.config_value, self.value)
                except (ValueError, TypeError) as e:
                    logger.warning("Failed to convert loaded value for '%s': %s, using original value", env_name, e)
                    self.value = self.config_value
            else:
                self.value = self.config_value
        else:
            logger.info("'%s' using default value: %s", env_name, env_value)
            self.value = env_value

        # 전역 레지스트리에 중복 확인 후 등록
        self._register_in_global_registry()

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

    def _register_in_global_registry(self):
        """
        전역 레지스트리에 중복 확인 후 등록
        같은 config_path를 가진 객체가 이미 있으면 기존 객체를 제거하고 새 객체를 등록
        """
        # 기존에 같은 config_path를 가진 객체가 있는지 확인
        existing_configs_to_remove = []
        for existing_config in PERSISTENT_CONFIG_REGISTRY:
            if existing_config.config_path == self.config_path:
                existing_configs_to_remove.append(existing_config)
                logger.debug("Found duplicate config_path '%s' in registry, will replace", self.config_path)

        # 중복된 객체들 제거
        for config_to_remove in existing_configs_to_remove:
            PERSISTENT_CONFIG_REGISTRY.remove(config_to_remove)
            logger.debug("Removed duplicate config from registry: %s", config_to_remove.env_name)

        # 새 객체 등록
        PERSISTENT_CONFIG_REGISTRY.append(self)
        logger.debug("Registered new config in registry: %s (config_path: %s)",
                    self.env_name, self.config_path)

    def update(self):
        """데이터베이스에서 최신 값을 다시 로드"""
        logger.info("Refreshing config '%s' from database...", self.env_name)
        new_value = get_config_value(self.config_path)
        if new_value is not None:
            self.value = new_value
            self.config_value = new_value
            logger.info("Config '%s' successfully refreshed from database: %s", self.env_name, self.value)
        else:
            logger.warning("No saved value found in database for '%s', keeping current value: %s", self.env_name, self.value)

    def save(self):
        """현재 값을 데이터베이스에 저장"""
        logger.info("Saving config '%s' to database: %s", self.env_name, self.value)
        set_config_value(self.config_path, self.value)
        self.config_value = self.value
        logger.info("Config '%s' successfully saved to database", self.env_name)

    def reset_to_default(self):
        """기본값으로 재설정"""
        logger.info("Resetting '%s' to default value: %s", self.env_name, self.env_value)
        self.value = self.env_value
        self.save()

    def refresh(self):
        """데이터베이스에서 최신 값을 다시 로드 (update의 별칭)"""
        return self.update()

# 유틸리티 함수들
def get_all_persistent_configs() -> List[PersistentConfig]:
    """등록된 모든 PersistentConfig 객체 반환"""
    return PERSISTENT_CONFIG_REGISTRY.copy()

def get_registry_statistics() -> Dict[str, Any]:
    """레지스트리 통계 정보 반환"""
    config_paths = [config.config_path for config in PERSISTENT_CONFIG_REGISTRY]
    env_names = [config.env_name for config in PERSISTENT_CONFIG_REGISTRY]

    # 중복 검사
    duplicate_paths = []
    duplicate_names = []

    seen_paths = set()
    seen_names = set()

    for path in config_paths:
        if path in seen_paths:
            duplicate_paths.append(path)
        seen_paths.add(path)

    for name in env_names:
        if name in seen_names:
            duplicate_names.append(name)
        seen_names.add(name)

    return {
        "total_configs": len(PERSISTENT_CONFIG_REGISTRY),
        "unique_config_paths": len(set(config_paths)),
        "unique_env_names": len(set(env_names)),
        "duplicate_config_paths": duplicate_paths,
        "duplicate_env_names": duplicate_names,
        "has_duplicates": len(duplicate_paths) > 0 or len(duplicate_names) > 0
    }

def is_json_fallback_enabled() -> bool:
    """JSON fallback이 활성화되어 있는지 확인"""
    return JSON_FALLBACK_ENABLED

def get_json_fallback_status() -> Dict[str, Any]:
    """JSON fallback 상태 정보 반환"""
    return {
        "json_fallback_enabled": JSON_FALLBACK_ENABLED,
        "config_file_path": CONFIG_DB_PATH,
        "environment_variable": "DISABLE_JSON_FALLBACK",
        "current_env_value": os.environ.get("DISABLE_JSON_FALLBACK", "not_set"),
        "description": "Set DISABLE_JSON_FALLBACK=true to disable JSON fallback when database connection fails"
    }

def refresh_all_configs():
    """모든 PersistentConfig 객체를 데이터베이스에서 다시 로드"""
    logger.info("Starting refresh of all persistent configs from database...")
    refreshed_count = 0

    for config in PERSISTENT_CONFIG_REGISTRY:
        try:
            config.update()
            refreshed_count += 1
        except Exception as e:
            logger.error("Failed to refresh config '%s': %s", config.env_name, e)

    logger.info("Completed refresh of %d/%d persistent configs from database", refreshed_count, len(PERSISTENT_CONFIG_REGISTRY))

def save_all_configs():
    """모든 PersistentConfig 객체를 데이터베이스에 저장"""
    logger.info("Starting save of all persistent configs to database...")
    saved_count = 0

    for config in PERSISTENT_CONFIG_REGISTRY:
        try:
            config.save()
            saved_count += 1
        except Exception as e:
            logger.error("Failed to save config '%s': %s", config.env_name, e)

    logger.info("Completed save of %d/%d persistent configs to database", saved_count, len(PERSISTENT_CONFIG_REGISTRY))

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

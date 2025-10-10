"""
Config Composer - 모든 설정을 통합 관리 (자동 발견 기능 포함)
"""
import os
import importlib
import logging
from typing import Dict, Any
from pathlib import Path
from config.base_config import BaseConfig
from config.persistent_config import PersistentConfig, export_config_summary, refresh_all_configs, save_all_configs
from service.database.database_manager import initialize_database
logger = logging.getLogger("config-composer")

class ConfigComposer:
    """
    모든 설정을 자동 발견하여 통합적으로 관리하는 클래스
    sub_config/ 디렉토리의 *_config.py 파일들을 자동으로 스캔하고 로드합니다.
    """

    def __init__(self):
        # 동적으로 로드된 설정 카테고리들을 저장
        self.config_categories: Dict[str, Any] = {}

        # 모든 설정을 저장하는 딕셔너리
        self.all_configs: Dict[str, PersistentConfig] = {}

        self.logger = logger

        # 설정 카테고리들을 자동으로 발견하고 로드
        self._discover_and_load_configs()

    def _discover_and_load_configs(self):
        """
        sub_config/ 디렉토리에서 *_config.py 파일들을 자동으로 발견하고 로드
        """
        sub_config_dir = Path(__file__).parent / "sub_config"

        # *_config.py 패턴의 파일들 찾기
        config_files = []
        for file in sub_config_dir.glob("*_config.py"):
            if file.name != "__init__.py":
                config_files.append(file)

        self.logger.info("Found %d config files: %s", len(config_files), [f.name for f in config_files])

        # 각 설정 파일을 동적으로 로드
        for config_file in config_files:
            try:
                # 파일명에서 카테고리명 추출 (예: openai_config.py -> openai)
                category_name = config_file.stem.replace("_config", "")

                # 모듈 이름 생성
                module_name = f"config.sub_config.{config_file.stem}"

                # 동적으로 모듈 import
                module = importlib.import_module(module_name)

                # 클래스 이름을 다양한 방식으로 시도
                possible_class_names = [
                    f"{category_name.title()}Config",           # OpenaiConfig
                    f"{category_name.upper()}Config",           # OPENAIConfig
                    f"{category_name.capitalize()}Config",      # OpenaiConfig
                    "".join(word.capitalize() for word in category_name.split("_")) + "Config"  # OpenAIConfig (if underscore separated)
                ]

                config_class = None
                for class_name in possible_class_names:
                    if hasattr(module, class_name):
                        config_class = getattr(module, class_name)
                        break

                if config_class is None:
                    # 마지막 시도: 모듈에서 BaseConfig를 상속받은 클래스 찾기
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if (isinstance(attr, type) and
                            issubclass(attr, BaseConfig) and
                            attr != BaseConfig):
                            config_class = attr
                            break

                if config_class is None:
                    raise AttributeError(f"No valid config class found in {module_name}")

                # 인스턴스 생성
                config_instance = config_class()

                # 카테고리로 저장
                self.config_categories[category_name] = config_instance

                # 동적 속성으로 설정 (self.openai, self.app 등)
                setattr(self, category_name, config_instance)

                self.logger.info("Successfully loaded config category: %s", category_name)

            except ImportError as e:
                self.logger.error("Failed to load config file %s: %s", config_file.name, e)

        self.logger.info("Auto-discovered %d config categories: %s", len(self.config_categories), list(self.config_categories.keys()))

    def initialize_all_configs(self) -> Dict[str, Any]:
        """
        모든 설정을 초기화하고 app.state에 저장할 데이터 구조 반환
        """
        try:
            self.logger.info("Initializing all configurations...")

            # 1. 데이터베이스 설정 우선 초기화
            if "database" in self.config_categories:
                database_configs = self.config_categories["database"].initialize()
                self.all_configs.update(database_configs)

                # 2. 데이터베이스 초기화 (마이그레이션 포함)
                db_initialized = initialize_database(self.config_categories["database"])
                if db_initialized:
                    self.logger.info("Database initialized successfully")
                else:
                    self.logger.warning("Database initialization failed, using JSON fallback")

            # 3. 나머지 설정 카테고리들 초기화
            for category_name, config_instance in self.config_categories.items():
                if category_name != "database":  # 데이터베이스는 이미 초기화됨
                    try:
                        category_configs = config_instance.initialize()
                        self.all_configs.update(category_configs)
                        self.logger.debug("Initialized %s config: %d settings", category_name, len(category_configs))
                    except AttributeError as e:
                        self.logger.error("Failed to initialize %s config: %s", category_name, e)

            # 4. VectorDB와 OpenAI 설정 간 연결 설정
            if "vectordb" in self.config_categories and "openai" in self.config_categories:
                self.config_categories["vectordb"].set_openai_config(self.config_categories["openai"])
                self.logger.info("Connected VectorDB config with OpenAI config")

                # 5. OpenAI API 키가 있으면 자동으로 최적의 제공자로 전환 시도
                if self.config_categories["openai"].API_KEY.value:
                    switched = self.config_categories["vectordb"].check_and_switch_to_best_provider()
                    if switched:
                        self.logger.info("Auto-switched to optimal embedding provider")

            # 6. 모든 config 값들을 DB에 초기값으로 저장 (첫 실행 시)
            self._ensure_initial_config_values_in_db()

            self.logger.info("Successfully initialized %d configurations", len(self.all_configs))

            # app.state에 저장할 구조화된 데이터 반환
            result = self.config_categories.copy()
            result["all_configs"] = self.all_configs
            return result

        except Exception as e:
            self.logger.error("Failed to initialize configurations: %s", e)
            raise

    def initialize_database_config_only(self) -> Dict[str, Any]:
        """
        데이터베이스 설정만 먼저 초기화 (테이블 생성을 위해)
        """
        try:
            self.logger.info("Initializing database configuration only...")

            if "database" in self.config_categories:
                database_configs = self.config_categories["database"].initialize()
                self.all_configs.update(database_configs)

                # 데이터베이스 연결만 설정 (마이그레이션은 하지 않음)
                db_initialized = initialize_database(self.config_categories["database"])
                if db_initialized:
                    self.logger.info("Database connection initialized successfully")
                    return self.config_categories["database"]
                else:
                    self.logger.warning("Database connection failed")

            return None

        except Exception as e:
            self.logger.error("Failed to initialize database configuration: %s", e)
            return None

    def initialize_remaining_configs(self) -> Dict[str, Any]:
        """
        데이터베이스 외의 나머지 설정들을 초기화 (DB 테이블이 생성된 후)
        """
        try:
            self.logger.info("Initializing remaining configurations...")

            # 나머지 설정 카테고리들 초기화
            for category_name, config_instance in self.config_categories.items():
                if category_name != "database":  # 데이터베이스는 이미 초기화됨
                    try:
                        category_configs = config_instance.initialize()
                        self.all_configs.update(category_configs)
                        self.logger.info("Initialized %s config: %d settings", category_name, len(category_configs))
                    except AttributeError as e:
                        self.logger.error("Failed to initialize %s config: %s", category_name, e)

            # 모든 config 값들을 DB에 초기값으로 저장 (첫 실행 시)
            self._ensure_initial_config_values_in_db()

            self.logger.info("Successfully initialized %d configurations", len(self.all_configs))

            # app.state에 저장할 구조화된 데이터 반환
            result = self.config_categories.copy()
            result["all_configs"] = self.all_configs
            return result

        except Exception as e:
            self.logger.error("Failed to initialize remaining configurations: %s", e)
            raise

    def get_all_config(self, **kwargs):
        result = self.config_categories.copy()
        result["all_configs"] = self.all_configs
        return result

    def get_config_by_name(self, config_name: str) -> PersistentConfig:
        """
        이름으로 특정 설정 가져오기
        """
        if config_name in self.all_configs:
            return self.all_configs[config_name]
        raise KeyError(f"Configuration '{config_name}' not found")

    def update_config_by_name(self, config_name: str, new_value: Any) -> None:
        """
        이름으로 특정 설정 업데이트
        """
        if config_name in self.all_configs:
            self.all_configs[config_name].value = new_value
            self.all_configs[config_name].save()
        else:
            raise KeyError(f"Configuration '{config_name}' not found")

    def get_config_by_category_name(self, category_name: str):
        """
        이름으로 특정 설정 가져오기
        """
        if category_name in self.config_categories:
            return self.config_categories[category_name]
        raise KeyError(f"Configuration '{category_name}' not found")

    def get_config_summary(self) -> Dict[str, Any]:
        """
        모든 설정의 요약 정보 반환 (동적 카테고리 지원)
        """
        categories_summary = {}
        for category_name, config_instance in self.config_categories.items():
            try:
                categories_summary[category_name] = config_instance.get_config_summary()
            except AttributeError as e:
                self.logger.error("Failed to get summary for %s: %s", category_name, e)
                categories_summary[category_name] = {"error": str(e)}

        return {
            "total_configs": len(self.all_configs),
            "discovered_categories": list(self.config_categories.keys()),
            "categories": categories_summary,
            "persistent_summary": export_config_summary()
        }

    def refresh_all(self):
        """
        모든 설정을 데이터베이스에서 다시 로드
        """
        self.logger.info("=== Starting database refresh for all configurations ===")
        refresh_all_configs()
        self.logger.info("=== Database refresh completed for all configurations ===")

    def save_all(self):
        """
        모든 설정을 데이터베이스에 저장
        """
        self.logger.info("=== Starting database save for all configurations ===")
        save_all_configs()
        self.logger.info("=== Database save completed for all configurations ===")

    def ensure_directories(self):
        """
        필요한 디렉토리들 생성 (고정된 디렉토리 목록)
        """
        # 고정된 디렉토리 목록
        required_directories = ["constants", "downloads"]

        try:
            model_storage_config = self.get_config_by_name("MODEL_STORAGE_DIRECTORY")
            if model_storage_config and model_storage_config.value:
                required_directories.append(model_storage_config.value)
        except KeyError:
            pass

        for directory in required_directories:
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                self.logger.info("Created directory: %s", directory)
            else:
                self.logger.info("Directory already exists: %s", directory)

    def validate_critical_configs(self) -> Dict[str, Any]:
        """
        중요한 설정들이 올바르게 설정되었는지 검증 (동적 카테고리 지원)
        """
        validation_results = {
            "valid": True,
            "warnings": [],
            "errors": []
        }

        # 포트 범위 검증 (app 카테고리가 있는 경우)
        if "app" in self.config_categories:
            app_config = self.config_categories["app"]
            if hasattr(app_config, "PORT"):
                port = app_config.PORT.value
                if not (1 <= port <= 65535):
                    validation_results["errors"].append(f"Invalid port number: {port}")
                    validation_results["valid"] = False

        return validation_results

    def _ensure_initial_config_values_in_db(self):
        """
        모든 config 값들을 DB에 초기값으로 저장 (첫 실행 시)
        이미 DB에 값이 있으면 건드리지 않고, 없는 값들만 저장
        """
        try:
            self.logger.info("Ensuring all initial config values are in database...")
            saved_count = 0

            # 모든 PersistentConfig 객체들을 확인
            for config_obj in self.all_configs.values():
                if isinstance(config_obj, PersistentConfig):
                    # DB에 값이 없고 현재 값이 기본값/env값인 경우에만 저장
                    if config_obj.config_value is None:
                        try:
                            config_obj.save()
                            saved_count += 1
                            # Suppress successful save logs for initial values
                        except (ImportError, AttributeError, ValueError) as e:
                            self.logger.warning("Failed to save initial value for %s: %s",
                                              config_obj.env_name, e)

            self.logger.info("Successfully ensured %d initial config values in database", saved_count)

        except Exception as e:
            self.logger.error("Failed to ensure initial config values in database: %s", e)
            raise

    def update_config(self, config_name: str, new_value: Any) -> Dict[str, Any]:
        """
        설정값을 업데이트하고 저장하는 통합 메서드

        Args:
            config_name: 설정 이름
            new_value: 새로운 값

        Returns:
            Dict: 업데이트 결과 정보

        Raises:
            KeyError: 설정이 존재하지 않는 경우
            ValueError: 타입 변환 실패 시
        """
        try:
            # 설정 객체 가져오기
            config_obj = self.get_config_by_name(config_name)
            old_value = config_obj.value

            # 타입별로 적절하게 변환 처리
            if isinstance(config_obj.env_value, bool):
                # boolean 타입 처리
                if isinstance(new_value, str):
                    # 문자열 형태의 boolean 처리
                    new_value_lower = new_value.lower().strip()
                    if new_value_lower in ['true', '1', 'yes', 'on']:
                        config_obj.value = True
                    elif new_value_lower in ['false', '0', 'no', 'off']:
                        config_obj.value = False
                    else:
                        raise ValueError(f"Cannot convert '{new_value}' to boolean")
                else:
                    config_obj.value = bool(new_value)
            elif isinstance(config_obj.env_value, int):
                # integer 타입 처리
                if isinstance(new_value, str):
                    new_value = new_value.strip()
                config_obj.value = int(new_value)
            elif isinstance(config_obj.env_value, float):
                # float 타입 처리
                if isinstance(new_value, str):
                    new_value = new_value.strip()
                config_obj.value = float(new_value)
            else:
                # 기본적으로 문자열로 처리
                config_obj.value = str(new_value)

            # 1. DB에 저장
            config_obj.save()

            # 2. config_composer의 all_configs 업데이트
            self.update_config_by_name(config_name, config_obj.value)

            # 3. 런타임에서 필요한 부수효과 적용 (환경변수 등)
            self._apply_runtime_side_effects(config_name, config_obj.value)

            self.logger.info("Successfully updated config '%s': %s -> %s",
                           config_name, old_value, config_obj.value)

            return {
                "config_name": config_name,
                "old_value": old_value,
                "new_value": config_obj.value,
                "success": True
            }

        except KeyError:
            self.logger.error("Config '%s' not found", config_name)
            raise KeyError(f"Config '{config_name}' not found")
        except (ValueError, TypeError) as e:
            self.logger.error("Failed to convert value for '%s': %s", config_name, e)
            raise ValueError(f"Invalid value type for '{config_name}': {e}")
        except Exception as e:
            self.logger.error("Failed to update config '%s': %s", config_name, e)
            raise

    def _apply_runtime_side_effects(self, config_name: str, new_value: Any) -> None:
        """특정 설정 변경 시 런타임 환경에 즉시 반영이 필요한 부수효과 처리"""
        if config_name == "OPENAI_API_KEY":
            sanitized = str(new_value).strip() if new_value is not None else ""
            if sanitized:
                os.environ["OPENAI_API_KEY"] = sanitized
                self.logger.debug("Updated OPENAI_API_KEY environment variable")
            else:
                removed = False
                if os.environ.pop("OPENAI_API_KEY", None) is not None:
                    removed = True
                if removed:
                    self.logger.debug("Cleared OPENAI_API_KEY environment variable")
        elif config_name == "OPENAI_API_BASE_URL":
            sanitized = str(new_value).strip() if new_value is not None else ""
            if sanitized:
                os.environ["OPENAI_API_BASE_URL"] = sanitized
                os.environ["OPENAI_BASE_URL"] = sanitized
                self.logger.debug("Updated OPENAI_API_BASE_URL/OPENAI_BASE_URL environment variables")
            else:
                removed = False
                if os.environ.pop("OPENAI_API_BASE_URL", None) is not None:
                    removed = True
                if os.environ.pop("OPENAI_BASE_URL", None) is not None:
                    removed = True
                if removed:
                    self.logger.debug("Cleared OPENAI_API_BASE_URL/OPENAI_BASE_URL environment variables")

# 전역 설정 컴포저 인스턴스
config_composer = ConfigComposer()

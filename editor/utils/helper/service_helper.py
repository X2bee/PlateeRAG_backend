import logging
from typing import Optional

logger = logging.getLogger(__name__)

def get_rag_service() -> Optional[object]:
    """FastAPI 앱에서 RAG 서비스를 가져오는 헬퍼 함수"""
    try:
        import sys

        # 먼저 main 모듈에서 찾기
        if 'main' in sys.modules:
            main_module = sys.modules['main']
            if hasattr(main_module, 'app') and hasattr(main_module.app, 'state'):
                if hasattr(main_module.app.state, 'rag_service'):
                    rag_service = main_module.app.state.rag_service
                    logger.info("main 모듈에서 RAG 서비스를 찾았습니다.")
                    return rag_service

        # 다른 모듈들에서 찾기
        for module_name, module in sys.modules.items():
            if hasattr(module, 'app'):
                app = getattr(module, 'app')
                if hasattr(app, 'state') and hasattr(app.state, 'rag_service'):
                    rag_service = app.state.rag_service
                    logger.info(f"{module_name} 모듈에서 RAG 서비스를 찾았습니다.")
                    return rag_service

        logger.warning("RAG 서비스를 찾을 수 없습니다. 서버가 실행되지 않았을 수 있습니다.")
        return None

    except Exception as e:
        logger.error(f"RAG 서비스 접근 중 오류: {e}")
        return None

def get_db_manager() -> Optional[object]:
    """FastAPI 앱에서 DB 매니저를 가져오는 헬퍼 함수"""
    try:
        import sys

        # 먼저 main 모듈에서 찾기
        if 'main' in sys.modules:
            main_module = sys.modules['main']
            if hasattr(main_module, 'app') and hasattr(main_module.app, 'state'):
                if hasattr(main_module.app.state, 'app_db'):
                    db_manager = main_module.app.state.app_db
                    logger.info("main 모듈에서 DB 매니저를 찾았습니다.")
                    return db_manager

        # 다른 모듈들에서 찾기
        for module_name, module in sys.modules.items():
            if hasattr(module, 'app'):
                app = getattr(module, 'app')
                if hasattr(app, 'state') and hasattr(app.state, 'app_db'):
                    db_manager = app.state.app_db
                    logger.info(f"{module_name} 모듈에서 DB 매니저를 찾았습니다.")
                    return db_manager

        logger.warning("DB 매니저를 찾을 수 없습니다. 서버가 실행되지 않았을 수 있습니다.")
        return None

    except Exception as e:
        logger.error(f"DB 매니저 접근 중 오류: {e}")
        return None

def get_config_composer() -> Optional[object]:
    """FastAPI 앱에서 Config Composer를 가져오는 헬퍼 함수"""
    try:
        import sys

        # 먼저 main 모듈에서 찾기
        if 'main' in sys.modules:
            main_module = sys.modules['main']
            if hasattr(main_module, 'app') and hasattr(main_module.app, 'state'):
                if hasattr(main_module.app.state, 'config_composer'):
                    config_composer = main_module.app.state.config_composer
                    logger.info("main 모듈에서 Config Composer를 찾았습니다.")
                    return config_composer

        # 다른 모듈들에서 찾기
        for module_name, module in sys.modules.items():
            if hasattr(module, 'app'):
                app = getattr(module, 'app')
                if hasattr(app, 'state') and hasattr(app.state, 'config_composer'):
                    config_composer = app.state.config_composer
                    logger.info(f"{module_name} 모듈에서 Config Composer를 찾았습니다.")
                    return config_composer

        logger.warning("Config Composer를 찾을 수 없습니다. 서버가 실행되지 않았을 수 있습니다.")
        return None

    except Exception as e:
        logger.error(f"Config Composer 접근 중 오류: {e}")
        return None

class RAGServiceManager:
    """RAG 서비스 매니저 클래스 (캐싱 기능 포함)"""
    _cached_service = None

    @classmethod
    def get_service(cls):
        """캐시된 RAG 서비스 반환 (없으면 새로 찾기)"""
        if cls._cached_service is None:
            cls._cached_service = get_rag_service()
        return cls._cached_service

    @classmethod
    def clear_cache(cls):
        """캐시 초기화"""
        cls._cached_service = None

class DBManager:
    """DB 매니저 클래스 (캐싱 기능 포함)"""
    _cached_manager = None

    @classmethod
    def get_manager(cls):
        """캐시된 DB 매니저 반환 (없으면 새로 찾기)"""
        if cls._cached_manager is None:
            cls._cached_manager = get_db_manager()
        return cls._cached_manager

    @classmethod
    def clear_cache(cls):
        """캐시 초기화"""
        cls._cached_manager = None

class ConfigComposerManager:
    """Config Composer 매니저 클래스 (캐싱 기능 포함)"""
    _cached_composer = None

    @classmethod
    def get_composer(cls):
        """캐시된 Config Composer 반환 (없으면 새로 찾기)"""
        if cls._cached_composer is None:
            cls._cached_composer = get_config_composer()
        return cls._cached_composer

    @classmethod
    def clear_cache(cls):
        """캐시 초기화"""
        cls._cached_composer = None

# 통합 매니저 클래스
class AppServiceManager:
    """앱 서비스들을 통합 관리하는 클래스"""

    @staticmethod
    def get_rag_service():
        """RAG 서비스 반환"""
        return RAGServiceManager.get_service()

    @staticmethod
    def get_db_manager():
        """DB 매니저 반환"""
        return DBManager.get_manager()

    @staticmethod
    def get_config_composer():
        """Config Composer 반환"""
        return ConfigComposerManager.get_composer()

    @staticmethod
    def clear_all_caches():
        """모든 캐시 초기화"""
        RAGServiceManager.clear_cache()
        DBManager.clear_cache()
        ConfigComposerManager.clear_cache()

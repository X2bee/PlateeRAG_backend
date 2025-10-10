import logging
from typing import Optional
from service.retrieval.rag_service import RAGService

logger = logging.getLogger(__name__)

def get_rag_service() -> Optional[object]:
    try:
        import sys
        app = None
        if 'main' in sys.modules:
            main_module = sys.modules['main']
            if hasattr(main_module, 'app'):
                app = main_module.app
                logger.info("main 모듈에서 app을 찾았습니다.")

        if app is None:
            for module_name, module in sys.modules.items():
                if hasattr(module, 'app'):
                    app = getattr(module, 'app')
                    logger.info(f"{module_name} 모듈에서 app을 찾았습니다.")
                    break

        if app is None or not hasattr(app, 'state'):
            logger.warning("FastAPI app을 찾을 수 없습니다. 서버가 실행되지 않았을 수 있습니다.")
            return None

        config_composer = getattr(app.state, 'config_composer', None)
        embedding_client = getattr(app.state, 'embedding_client', None)
        vector_manager = getattr(app.state, 'vector_manager', None)
        document_processor = getattr(app.state, 'document_processor', None)
        document_info_generator = getattr(app.state, 'document_info_generator', None)

        if not all([config_composer, embedding_client, vector_manager, document_processor, document_info_generator]):
            logger.warning("RAG 서비스 생성에 필요한 일부 서비스를 찾을 수 없습니다.")
            return None

        rag_service = RAGService(
            config_composer,
            embedding_client,
            vector_manager,
            document_processor,
            document_info_generator
        )
        logger.info("RAG 서비스를 성공적으로 생성했습니다.")
        return rag_service

    except Exception as e:
        logger.error(f"RAG 서비스 생성 중 오류: {e}")
        return None

def get_db_manager() -> Optional[object]:
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

def get_guarder_service() -> Optional[object]:
    """FastAPI 앱에서 Guarder 서비스를 가져오는 헬퍼 함수"""
    try:
        import sys

        # 먼저 main 모듈에서 찾기
        if 'main' in sys.modules:
            main_module = sys.modules['main']
            try:
                if hasattr(main_module, 'app'):
                    app = main_module.app
                    if hasattr(app, 'state'):
                        state = app.state
                        if hasattr(state, 'guarder_service'):
                            guarder_service = getattr(state, 'guarder_service', None)
                            if guarder_service is not None:
                                logger.info("main 모듈에서 Guarder 서비스를 찾았습니다.")
                                return guarder_service
            except Exception as app_error:
                logger.debug(f"main 모듈에서 app.state 접근 실패: {app_error}")

        # 다른 모듈들에서 찾기
        for module_name, module in sys.modules.items():
            try:
                if hasattr(module, 'app'):
                    app = getattr(module, 'app')
                    if hasattr(app, 'state'):
                        state = app.state
                        if hasattr(state, 'guarder_service'):
                            guarder_service = getattr(state, 'guarder_service', None)
                            if guarder_service is not None:
                                logger.info(f"{module_name} 모듈에서 Guarder 서비스를 찾았습니다.")
                                return guarder_service
            except Exception as app_error:
                logger.debug(f"{module_name} 모듈에서 app.state 접근 실패: {app_error}")
                continue

        logger.warning("Guarder 서비스를 찾을 수 없습니다. 서버가 실행되지 않았을 수 있습니다.")
        return None

    except Exception as e:
        logger.error(f"Guarder 서비스 접근 중 오류: {e}")
        return None


def get_mlflow_artifact_service() -> Optional[object]:
    """FastAPI 앱에서 MLflow artifact 서비스를 가져오는 헬퍼 함수"""
    try:
        import sys

        if 'main' in sys.modules:
            main_module = sys.modules['main']
            try:
                if hasattr(main_module, 'app'):
                    app = main_module.app
                    if hasattr(app, 'state') and hasattr(app.state, 'mlflow_service'):
                        service = getattr(app.state, 'mlflow_service', None)
                        if service is not None:
                            logger.info("main 모듈에서 MLflow 서비스를 찾았습니다.")
                            return service
            except Exception as app_error:
                logger.debug(f"main 모듈에서 MLflow 서비스 접근 실패: {app_error}")

        for module_name, module in sys.modules.items():
            try:
                if hasattr(module, 'app'):
                    app = getattr(module, 'app')
                    if hasattr(app, 'state') and hasattr(app.state, 'mlflow_service'):
                        service = getattr(app.state, 'mlflow_service', None)
                        if service is not None:
                            logger.info(f"{module_name} 모듈에서 MLflow 서비스를 찾았습니다.")
                            return service
            except Exception as app_error:
                logger.debug(f"{module_name} 모듈에서 MLflow 서비스 접근 실패: {app_error}")
                continue

        logger.warning("MLflow 서비스를 찾을 수 없습니다. 서버가 실행되지 않았을 수 있습니다.")
        return None

    except Exception as exc:
        logger.error(f"MLflow 서비스 접근 중 오류: {exc}")
        return None

class RAGServiceManager:
    @classmethod
    def get_service(cls):
        cls._cached_service = get_rag_service()
        return cls._cached_service

    @classmethod
    def clear_cache(cls):
        """캐시 초기화"""
        cls._cached_service = None

class DBManager:
    @classmethod
    def get_manager(cls):
        cls._cached_manager = get_db_manager()
        return cls._cached_manager

    @classmethod
    def clear_cache(cls):
        """캐시 초기화"""
        cls._cached_manager = None

class ConfigComposerManager:
    @classmethod
    def get_composer(cls):
        cls._cached_composer = get_config_composer()
        return cls._cached_composer

    @classmethod
    def clear_cache(cls):
        """캐시 초기화"""
        cls._cached_composer = None

class GuarderServiceManager:
    @classmethod
    def get_service(cls):
        cls._cached_service = get_guarder_service()
        return cls._cached_service

    @classmethod
    def clear_cache(cls):
        """캐시 초기화"""
        cls._cached_service = None


class MLflowServiceManager:
    @classmethod
    def get_service(cls):
        cls._cached_service = get_mlflow_artifact_service()
        return cls._cached_service

    @classmethod
    def clear_cache(cls):
        """캐시 초기화"""
        cls._cached_service = None

# 통합 매니저 클래스
class AppServiceManager:
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
    def get_guarder_service():
        """Guarder 서비스 반환"""
        return GuarderServiceManager.get_service()

    @staticmethod
    def get_mlflow_service():
        """MLflow artifact 서비스 반환"""
        return MLflowServiceManager.get_service()

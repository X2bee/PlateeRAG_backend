from fastapi import HTTPException, Request
from service.database import AppDatabaseManager
from config.config_composer import ConfigComposer
from service.vast.vast_service import VastService
from service.vast.proxy_client import VastProxyClient
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from service.vector_db.vector_manager import VectorManager
    from service.retrieval.document_processor.document_processor import DocumentProcessor
    from service.retrieval.document_info_generator.document_info_generator import DocumentInfoGenerator
    from service.retrieval.rag_service import RAGService
    from service.database.performance_controller_helper import PerformanceControllerHelper
    from service.stt.huggingface_stt import HuggingFaceSTT
    from service.data_manager.data_manager_register import DataManagerRegistry
    from service.mlflow.mlflow_artifact_service import MLflowArtifactService

def get_db_manager(request: Request) -> AppDatabaseManager:
    """데이터베이스 매니저 의존성 주입"""
    if hasattr(request.app.state, 'app_db') and request.app.state.app_db:
        return request.app.state.app_db
    else:
        raise HTTPException(status_code=500, detail="Database connection not available")

def get_config_composer(request: Request) -> ConfigComposer:
    """request.app.state에서 config_composer 가져오기"""
    if hasattr(request.app.state, 'config_composer') and request.app.state.config_composer:
        return request.app.state.config_composer
    else:
        from config.config_composer import config_composer
        return config_composer

def get_embedding_client(request: Request):
    """임베딩 클라이언트 의존성 주입"""
    if hasattr(request.app.state, 'embedding_client') and request.app.state.embedding_client:
        return request.app.state.embedding_client
    else:
        raise HTTPException(status_code=500, detail="Embedding client not available")

def get_vector_manager(request: Request) -> 'VectorManager':
    """벡터 매니저 의존성 주입"""
    if hasattr(request.app.state, 'vector_manager') and request.app.state.vector_manager:
        return request.app.state.vector_manager
    else:
        raise HTTPException(status_code=500, detail="Vector manager not available")

def get_document_processor(request: Request) -> 'DocumentProcessor':
    """문서 처리기 의존성 주입"""
    if hasattr(request.app.state, 'document_processor') and request.app.state.document_processor:
        return request.app.state.document_processor
    else:
        raise HTTPException(status_code=500, detail="Document processor not available")

def get_document_info_generator(request: Request) -> 'DocumentInfoGenerator':
    """문서 정보 생성기 의존성 주입"""
    if hasattr(request.app.state, 'document_info_generator') and request.app.state.document_info_generator:
        return request.app.state.document_info_generator
    else:
        raise HTTPException(status_code=500, detail="Document info generator not available")

def get_vast_service(request: Request) -> VastService:
    """VAST 서비스 의존성 주입"""
    if hasattr(request.app.state, 'vast_service') and request.app.state.vast_service:
        return request.app.state.vast_service
    else:
        raise HTTPException(status_code=500, detail="VAST service not available")


def get_vast_proxy_client(request: Request) -> VastProxyClient:
    """Vast proxy client 의존성 주입"""
    if hasattr(request.app.state, 'vast_proxy_client') and request.app.state.vast_proxy_client:
        return request.app.state.vast_proxy_client
    if hasattr(request.app.state, 'vast_service') and isinstance(request.app.state.vast_service, VastProxyClient):
        return request.app.state.vast_service
    raise HTTPException(status_code=500, detail="VAST proxy client not available")

def get_stt_service(request: Request) -> 'HuggingFaceSTT':
    """STT 서비스 의존성 주입"""
    if hasattr(request.app.state, 'stt_service') and request.app.state.stt_service:
        return request.app.state.stt_service
    else:
        raise HTTPException(status_code=500, detail="STT service not available")

def get_tts_service(request: Request):
    """TTS 서비스 의존성 주입"""
    if hasattr(request.app.state, 'tts_service') and request.app.state.tts_service:
        return request.app.state.tts_service
    else:
        raise HTTPException(status_code=500, detail="TTS service not available")

def get_performance_controller(request: Request) -> 'PerformanceControllerHelper':
    app_db = get_db_manager(request)
    try:
        from service.database.performance_controller_helper import PerformanceControllerHelper
    except ModuleNotFoundError as exc:
        raise HTTPException(status_code=500, detail="Performance controller not available") from exc

    return PerformanceControllerHelper(app_db)

def get_rag_service(request: Request) -> 'RAGService':
    try:
        from service.retrieval.rag_service import RAGService
    except ModuleNotFoundError as exc:
        raise HTTPException(status_code=500, detail="RAG service not available") from exc

    config_composer = get_config_composer(request)
    embedding_client = get_embedding_client(request)
    vector_manager = get_vector_manager(request)
    document_processor = get_document_processor(request)
    document_info_generator = get_document_info_generator(request)

    rag_service = RAGService(config_composer, embedding_client, vector_manager, document_processor, document_info_generator)

    if rag_service:
        return rag_service
    else:
        raise HTTPException(status_code=500, detail="RAG service not available")

def get_data_manager_registry(request: Request) -> 'DataManagerRegistry':
    """request에서 DataManagerRegistry 가져오기"""
    if hasattr(request.app.state, 'data_manager_registry') and request.app.state.data_manager_registry:
        return request.app.state.data_manager_registry
    else:
        raise HTTPException(status_code=500, detail="DataManagerRegistry가 초기화되지 않았습니다")


def get_mlflow_service(request: Request) -> 'MLflowArtifactService':
    """MLflow artifact service dependency"""
    if hasattr(request.app.state, 'mlflow_service') and request.app.state.mlflow_service:
        return request.app.state.mlflow_service
    raise HTTPException(status_code=500, detail="MLflow service not available")

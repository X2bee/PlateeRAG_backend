from fastapi import HTTPException, Request
from service.vast.vast_service import VastService

def get_config_composer(request: Request):
    """request.app.state에서 config_composer 가져오기"""
    if hasattr(request.app.state, 'config_composer') and request.app.state.config_composer:
        return request.app.state.config_composer
    else:
        from config.config_composer import config_composer
        return config_composer

def get_embedding_client(request: Request):
    """임베딩 클라이언트 의존성 주입"""
    if hasattr(request.app.state, 'rag_service') and request.app.state.rag_service:
        return request.app.state.rag_service.embeddings_client
    else:
        raise HTTPException(status_code=500, detail="Embedding client not available")

def get_vector_manager(request: Request):
    """벡터 매니저 의존성 주입"""
    if hasattr(request.app.state, 'rag_service') and request.app.state.rag_service:
        return request.app.state.rag_service.vector_manager
    else:
        raise HTTPException(status_code=500, detail="Vector manager not available")

def get_rag_service(request: Request):
    """RAG 서비스 의존성 주입"""
    if hasattr(request.app.state, 'rag_service') and request.app.state.rag_service:
        return request.app.state.rag_service
    else:
        raise HTTPException(status_code=500, detail="RAG service not available")

def get_document_processor(request: Request):
    """문서 처리기 의존성 주입"""
    if hasattr(request.app.state, 'document_processor') and request.app.state.document_processor:
        return request.app.state.document_processor
    else:
        raise HTTPException(status_code=500, detail="Document processor not available")

def get_db_manager(request: Request):
    """데이터베이스 매니저 의존성 주입"""
    if hasattr(request.app.state, 'app_db') and request.app.state.app_db:
        return request.app.state.app_db
    else:
        raise HTTPException(status_code=500, detail="Database connection not available")

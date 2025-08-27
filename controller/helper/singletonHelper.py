from fastapi import HTTPException, Request
from service.retrieval.rag_service import RAGService

def get_db_manager(request: Request):
    """데이터베이스 매니저 의존성 주입"""
    if hasattr(request.app.state, 'app_db') and request.app.state.app_db:
        return request.app.state.app_db
    else:
        raise HTTPException(status_code=500, detail="Database connection not available")

def get_config_composer(request: Request):
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

def get_vector_manager(request: Request):
    """벡터 매니저 의존성 주입"""
    if hasattr(request.app.state, 'vector_manager') and request.app.state.vector_manager:
        return request.app.state.vector_manager
    else:
        raise HTTPException(status_code=500, detail="Vector manager not available")

def get_document_processor(request: Request):
    """문서 처리기 의존성 주입"""
    if hasattr(request.app.state, 'document_processor') and request.app.state.document_processor:
        return request.app.state.document_processor
    else:
        raise HTTPException(status_code=500, detail="Document processor not available")

def get_document_info_generator(request: Request):
    """문서 정보 생성기 의존성 주입"""
    if hasattr(request.app.state, 'document_info_generator') and request.app.state.document_info_generator:
        return request.app.state.document_info_generator
    else:
        raise HTTPException(status_code=500, detail="Document info generator not available")

def get_rag_service(request: Request):
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

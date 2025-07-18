"""
RAG 컨트롤러

RAG(Retrieval-Augmented Generation) API 엔드포인트를 제공합니다.
비즈니스 로직은 src.rag 모듈들에 위임하여 관심사를 분리합니다.
"""

from fastapi import APIRouter, HTTPException, Request, UploadFile, File, Form
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union
import logging
import aiofiles
import os
import json
from pathlib import Path

from service.rag import RAGService, EmbeddingFactory

logger = logging.getLogger("rag-controller")
router = APIRouter(prefix="/rag", tags=["rag"])

# Pydantic Models
class VectorPoint(BaseModel):
    id: Optional[Union[str, int]] = None
    vector: List[float]
    payload: Optional[Dict[str, Any]] = None

class SearchQuery(BaseModel):
    vector: List[float]
    limit: int = 10
    score_threshold: Optional[float] = None
    filter: Optional[Dict[str, Any]] = None

class CollectionCreateRequest(BaseModel):
    collection_name: str
    vector_size: int
    distance: str = "Cosine"
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class CollectionDeleteRequest(BaseModel):
    collection_name: str

class InsertPointsRequest(BaseModel):
    collection_name: str
    points: List[VectorPoint]

class SearchRequest(BaseModel):
    collection_name: str
    query: SearchQuery

class DeletePointsRequest(BaseModel):
    collection_name: str
    point_ids: List[Union[str, int]]

class DocumentSearchRequest(BaseModel):
    collection_name: str
    query_text: str
    limit: int = 5
    score_threshold: Optional[float] = 0.7
    filter: Optional[Dict[str, Any]] = None

class EmbeddingProviderSwitchRequest(BaseModel):
    new_provider: str

class EmbeddingTestRequest(BaseModel):
    query_text: str = "Hello, world!"

def get_rag_service(request: Request) -> RAGService:
    """RAG 서비스 의존성 주입"""
    if hasattr(request.app.state, 'config') and request.app.state.config:
        return RAGService(request.app.state.config["vectordb"], request.app.state.config.get("openai"))
    else:
        raise HTTPException(status_code=500, detail="Configuration not available")

# Health & Status Endpoints
@router.get("/health")
async def health_check(request: Request):
    """RAG 시스템 연결 상태 확인"""
    try:
        rag_service = get_rag_service(request)
        
        health_status = {
            "qdrant_client": rag_service.vector_manager.is_connected(),
            "embeddings_client": bool(rag_service.embeddings_client),
            "embedding_provider": rag_service.config.EMBEDDING_PROVIDER.value
        }
        
        if rag_service.vector_manager.is_connected():
            collections = rag_service.vector_manager.list_collections()
            health_status.update({
                "collections_count": len(collections["collections"]),
                "qdrant_status": "connected"
            })
        else:
            health_status["qdrant_status"] = "disconnected"
        
        # 임베딩 클라이언트 상태 상세 확인
        if rag_service.embeddings_client:
            try:
                is_available = await rag_service.embeddings_client.is_available()
                health_status["embeddings_status"] = "available" if is_available else "unavailable"
                health_status["embeddings_available"] = is_available
            except Exception as e:
                health_status["embeddings_status"] = "error"
                health_status["embeddings_error"] = str(e)
                health_status["embeddings_available"] = False
        else:
            health_status["embeddings_status"] = "not_initialized"
            health_status["embeddings_available"] = False
        
        overall_status = "healthy" if all([
            rag_service.vector_manager.is_connected(), 
            rag_service.embeddings_client,
            health_status.get("embeddings_available", False)
        ]) else "partial"
        
        return {
            "status": overall_status,
            "message": "RAG system status check",
            "components": health_status
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "message": f"Health check failed: {e}"
        }

# Collection Management Endpoints
@router.get("/collections")
async def list_collections(request: Request):
    """모든 컬렉션 목록 조회"""
    try:
        rag_service = get_rag_service(request)
        return rag_service.vector_manager.list_collections()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list collections: {str(e)}")

@router.post("/collections")
async def create_collection(request: Request, collection_request: CollectionCreateRequest):
    """새 컬렉션 생성"""
    try:
        rag_service = get_rag_service(request)
        return rag_service.vector_manager.create_collection(
            collection_request.collection_name,
            collection_request.vector_size,
            collection_request.distance,
            collection_request.description,
            collection_request.metadata
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create collection: {str(e)}")

@router.delete("/collections")
async def delete_collection(request: Request, collection_request: CollectionDeleteRequest):
    """컬렉션 삭제"""
    try:
        rag_service = get_rag_service(request)
        return rag_service.vector_manager.delete_collection(collection_request.collection_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete collection: {str(e)}")

@router.get("/collections/{collection_name}")
async def get_collection_info(request: Request, collection_name: str):
    """특정 컬렉션 정보 조회"""
    try:
        rag_service = get_rag_service(request)
        return rag_service.vector_manager.get_collection_info(collection_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get collection info: {str(e)}")

# Document Management Endpoints
@router.post("/documents/upload")
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    collection_name: str = Form(...),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200),
    process_chunks: bool = Form(True),
    metadata: Optional[str] = Form(None)
):
    """문서 업로드 및 처리"""
    try:
        rag_service = get_rag_service(request)
        
        # 파일 저장
        upload_dir = Path("downloads")
        upload_dir.mkdir(exist_ok=True)
        
        file_path = upload_dir / file.filename
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        # 메타데이터 파싱
        doc_metadata = {}
        if metadata:
            try:
                doc_metadata = json.loads(metadata)
            except json.JSONDecodeError:
                logger.warning(f"Invalid metadata JSON: {metadata}")
        
        # 문서 처리
        result = await rag_service.process_document(
            str(file_path),
            collection_name,
            chunk_size,
            chunk_overlap,
            doc_metadata
        )
        
        # 임시 파일 삭제
        try:
            os.unlink(file_path)
        except Exception:
            pass
        
        return result
        
    except Exception as e:
        logger.error(f"Document upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload document: {str(e)}")

@router.post("/documents/search")
async def search_documents(request: Request, search_request: DocumentSearchRequest):
    """문서 검색"""
    try:
        rag_service = get_rag_service(request)
        result = await rag_service.search_documents(
            search_request.collection_name,
            search_request.query_text,
            search_request.limit,
            search_request.score_threshold,
            search_request.filter
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to search documents: {str(e)}")

@router.get("/collections/{collection_name}/documents")
async def list_documents_in_collection(request: Request, collection_name: str):
    """컬렉션 내 모든 문서 목록 조회"""
    try:
        rag_service = get_rag_service(request)
        return await rag_service.list_documents_in_collection(collection_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")

@router.get("/collections/{collection_name}/documents/{document_id}")
async def get_document_details(request: Request, collection_name: str, document_id: str):
    """특정 문서의 상세 정보 조회"""
    try:
        rag_service = get_rag_service(request)
        return rag_service.get_document_details(collection_name, document_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get document details: {str(e)}")

@router.delete("/collections/{collection_name}/documents/{document_id}")
async def delete_document_from_collection(request: Request, collection_name: str, document_id: str):
    """컬렉션에서 특정 문서 삭제"""
    try:
        rag_service = get_rag_service(request)
        return rag_service.delete_document_from_collection(collection_name, document_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")

# Legacy Vector DB Endpoints (for backwards compatibility)
@router.post("/points")
async def insert_points(request: Request, insert_request: InsertPointsRequest):
    """벡터 포인트 삽입 (기존 호환성)"""
    try:
        rag_service = get_rag_service(request)
        # VectorPoint 모델을 딕셔너리로 변환
        points_data = []
        for point in insert_request.points:
            point_dict = {
                "vector": point.vector,
                "payload": point.payload or {}
            }
            if point.id is not None:
                point_dict["id"] = point.id
            points_data.append(point_dict)
        
        return rag_service.vector_manager.insert_points(
            insert_request.collection_name,
            points_data
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to insert points: {str(e)}")

@router.post("/search")
async def search_points(request: Request, search_request: SearchRequest):
    """벡터 유사도 검색 (기존 호환성)"""
    try:
        rag_service = get_rag_service(request)
        return rag_service.vector_manager.search_points(
            search_request.collection_name,
            search_request.query.vector,
            search_request.query.limit,
            search_request.query.score_threshold,
            search_request.query.filter
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to search points: {str(e)}")

@router.delete("/points")
async def delete_points(request: Request, delete_request: DeletePointsRequest):
    """포인트 삭제 (기존 호환성)"""
    try:
        rag_service = get_rag_service(request)
        return rag_service.vector_manager.delete_points(
            delete_request.collection_name,
            delete_request.point_ids
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete points: {str(e)}")

# Configuration Endpoints
@router.get("/config")
async def get_rag_config(request: Request):
    """현재 RAG 시스템 설정 조회"""
    try:
        if hasattr(request.app.state, 'config') and request.app.state.config:
            vectordb_config = request.app.state.config["vectordb"]
            
            return {
                "vectordb": {
                    "host": vectordb_config.QDRANT_HOST.value,
                    "port": vectordb_config.QDRANT_PORT.value,
                    "use_grpc": vectordb_config.QDRANT_USE_GRPC.value,
                    "grpc_port": vectordb_config.QDRANT_GRPC_PORT.value,
                    "collection_name": vectordb_config.COLLECTION_NAME.value,
                    "vector_dimension": vectordb_config.VECTOR_DIMENSION.value,
                    "replicas": vectordb_config.REPLICAS.value,
                    "shards": vectordb_config.SHARDS.value
                },
                "embedding": {
                    "provider": vectordb_config.EMBEDDING_PROVIDER.value,
                    "auto_detect_dimension": vectordb_config.AUTO_DETECT_EMBEDDING_DIM.value,
                    "openai": {
                        "api_key_configured": bool(vectordb_config.get_openai_api_key()),
                        "model": vectordb_config.OPENAI_EMBEDDING_MODEL.value
                    },
                    "huggingface": {
                        "model_name": vectordb_config.HUGGINGFACE_MODEL_NAME.value,
                        "api_key_configured": bool(vectordb_config.HUGGINGFACE_API_KEY.value)
                    },
                    "custom_http": {
                        "url": vectordb_config.CUSTOM_EMBEDDING_URL.value,
                        "model": vectordb_config.CUSTOM_EMBEDDING_MODEL.value,
                        "api_key_configured": bool(vectordb_config.CUSTOM_EMBEDDING_API_KEY.value)
                    }
                }
            }
        else:
            raise HTTPException(status_code=500, detail="Configuration not available")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get config: {str(e)}")

# Embedding Management Endpoints
@router.get("/embedding/providers")
async def get_embedding_providers():
    """사용 가능한 임베딩 제공자 목록 조회"""
    try:
        providers = EmbeddingFactory.get_available_providers()
        return {
            "providers": providers,
            "total": len(providers)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get providers: {str(e)}")

@router.get("/embedding/test")
async def test_embedding_providers(request: Request):
    """모든 임베딩 제공자 테스트"""
    try:
        rag_service = get_rag_service(request)
        results = await EmbeddingFactory.test_all_providers(rag_service.config)
        return {
            "test_results": results,
            "current_provider": rag_service.config.EMBEDDING_PROVIDER.value
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to test providers: {str(e)}")

@router.get("/embedding/status")
async def get_embedding_status(request: Request):
    """현재 임베딩 클라이언트 상태 조회"""
    try:
        rag_service = get_rag_service(request)
        status_info = rag_service.get_embedding_status()
        
        # 실제 사용 가능성 체크 (비동기)
        if rag_service.embeddings_client:
            try:
                is_available = await rag_service.embeddings_client.is_available()
                status_info["available"] = is_available
            except Exception as e:
                status_info["available"] = False
                status_info["availability_error"] = str(e)
        
        return status_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get embedding status: {str(e)}")

@router.post("/embedding/test-query")
async def test_embedding_query(request: Request, test_request: EmbeddingTestRequest):
    """임베딩 생성 테스트"""
    try:
        rag_service = get_rag_service(request)
        
        # 자동 복구 로직이 포함된 임베딩 생성
        embedding = await rag_service.generate_query_embedding(test_request.query_text)
        
        return {
            "query_text": test_request.query_text,
            "embedding_dimension": len(embedding),
            "embedding_preview": embedding[:5],  # 처음 5개 값만 미리보기
            "provider": rag_service.config.EMBEDDING_PROVIDER.value,
            "success": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to test embedding: {str(e)}")

@router.post("/embedding/reload")
async def reload_embedding_client(request: Request):
    """임베딩 클라이언트 강제 재로드"""
    try:
        rag_service = get_rag_service(request)
        
        success = await rag_service.reload_embeddings_client()
        
        if success:
            provider_info = rag_service.embeddings_client.get_provider_info()
            return {
                "success": True,
                "message": "Embedding client reloaded successfully",
                "provider": rag_service.config.EMBEDDING_PROVIDER.value,
                "provider_info": provider_info
            }
        else:
            return {
                "success": False,
                "message": "Failed to reload embedding client",
                "provider": rag_service.config.EMBEDDING_PROVIDER.value
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload embedding client: {str(e)}")

@router.post("/embedding/switch-provider")
async def switch_embedding_provider(request: Request, switch_request: EmbeddingProviderSwitchRequest):
    """임베딩 제공자 변경"""
    try:
        rag_service = get_rag_service(request)
        new_provider = switch_request.new_provider
        
        # OpenAI 설정과 VectorDB 설정 연결 확인 및 재설정
        if hasattr(request.app.state, 'config') and request.app.state.config:
            app_config = request.app.state.config
            if "vectordb" in app_config and "openai" in app_config:
                app_config["vectordb"].set_openai_config(app_config["openai"])
        
        # 제공자 변경 시도
        old_provider = rag_service.config.EMBEDDING_PROVIDER.value
        success = rag_service.config.switch_embedding_provider(new_provider)
        
        if not success:
            # 실패 원인을 더 자세히 분석
            error_msg = f"Cannot switch to provider '{new_provider}'. "
            if new_provider.lower() == "openai":
                openai_key = rag_service.config.get_openai_api_key()
                if not openai_key:
                    error_msg += "OpenAI API key is not configured."
                else:
                    error_msg += "OpenAI configuration validation failed."
            elif new_provider.lower() == "custom_http":
                custom_url = rag_service.config.CUSTOM_EMBEDDING_URL.value
                if not custom_url or custom_url == "http://localhost:8000/v1":
                    error_msg += "Custom HTTP URL is not properly configured."
            
            raise HTTPException(status_code=400, detail=error_msg)
        
        # 클라이언트 재로드
        reload_success = await rag_service.reload_embeddings_client()
        
        if reload_success:
            provider_info = rag_service.embeddings_client.get_provider_info()
            return {
                "success": True,
                "message": f"Embedding provider switched from {old_provider} to {new_provider}",
                "old_provider": old_provider,
                "new_provider": new_provider,
                "provider_info": provider_info
            }
        else:
            # 실패한 경우 원래 제공자로 롤백
            rag_service.config.switch_embedding_provider(old_provider)
            await rag_service.reload_embeddings_client()
            
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to initialize new provider '{new_provider}'. Rolled back to '{old_provider}'"
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to switch embedding provider: {str(e)}")

@router.post("/embedding/auto-switch")
async def auto_switch_embedding_provider(request: Request):
    """자동으로 최적의 임베딩 제공자로 전환"""
    try:
        rag_service = get_rag_service(request)
        
        # OpenAI 설정과 VectorDB 설정 연결 확인 및 재설정
        if hasattr(request.app.state, 'config') and request.app.state.config:
            app_config = request.app.state.config
            if "vectordb" in app_config and "openai" in app_config:
                app_config["vectordb"].set_openai_config(app_config["openai"])
        
        old_provider = rag_service.config.EMBEDDING_PROVIDER.value
        
        # 최적의 제공자로 자동 전환 시도
        switched = rag_service.config.check_and_switch_to_best_provider()
        
        if switched:
            new_provider = rag_service.config.EMBEDDING_PROVIDER.value
            
            # 클라이언트 재로드
            reload_success = await rag_service.reload_embeddings_client()
            
            if reload_success:
                provider_info = rag_service.embeddings_client.get_provider_info()
                return {
                    "success": True,
                    "switched": True,
                    "message": f"Auto-switched embedding provider from {old_provider} to {new_provider}",
                    "old_provider": old_provider,
                    "new_provider": new_provider,
                    "provider_info": provider_info
                }
            else:
                # 실패한 경우 원래 제공자로 롤백
                rag_service.config.switch_embedding_provider(old_provider)
                await rag_service.reload_embeddings_client()
                
                return {
                    "success": False,
                    "switched": False,
                    "message": f"Failed to initialize new provider '{new_provider}'. Rolled back to '{old_provider}'",
                    "old_provider": old_provider,
                    "new_provider": old_provider
                }
        else:
            return {
                "success": True,
                "switched": False,
                "message": f"Current provider '{old_provider}' is already optimal",
                "current_provider": old_provider
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to auto-switch embedding provider: {str(e)}")

@router.get("/embedding/config-status")
async def get_embedding_config_status(request: Request):
    """임베딩 설정 상태 조회"""
    try:
        rag_service = get_rag_service(request)
        config_status = rag_service.config.get_embedding_provider_status()
        
        # 현재 클라이언트 상태 추가
        if rag_service.embeddings_client:
            try:
                is_available = await rag_service.embeddings_client.is_available()
                provider_info = rag_service.embeddings_client.get_provider_info()
                config_status.update({
                    "client_initialized": True,
                    "client_available": is_available,
                    "provider_info": provider_info
                })
            except Exception as e:
                config_status.update({
                    "client_initialized": True,
                    "client_available": False,
                    "client_error": str(e)
                })
        else:
            config_status.update({
                "client_initialized": False,
                "client_available": False
            })
        
        return config_status
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get config status: {str(e)}")

@router.get("/debug/embedding-info")
async def get_embedding_debug_info(request: Request):
    """디버깅을 위한 임베딩 상세 정보 조회"""
    try:
        rag_service = get_rag_service(request)
        vectordb_config = rag_service.config
        
        debug_info = {
            "current_provider": vectordb_config.EMBEDDING_PROVIDER.value,
            "auto_detect_dimension": vectordb_config.AUTO_DETECT_EMBEDDING_DIM.value,
            "vector_dimension": vectordb_config.VECTOR_DIMENSION.value,
            "client_initialized": bool(rag_service.embeddings_client),
            "dependencies": {},
            "provider_configs": {
                "openai": {
                    "api_key_configured": bool(vectordb_config.get_openai_api_key()),
                    "api_key_length": len(vectordb_config.get_openai_api_key()) if vectordb_config.get_openai_api_key() else 0,
                    "model": vectordb_config.OPENAI_EMBEDDING_MODEL.value
                },
                "huggingface": {
                    "model_name": vectordb_config.HUGGINGFACE_MODEL_NAME.value,
                    "api_key_configured": bool(vectordb_config.HUGGINGFACE_API_KEY.value)
                },
                "custom_http": {
                    "url": vectordb_config.CUSTOM_EMBEDDING_URL.value,
                    "model": vectordb_config.CUSTOM_EMBEDDING_MODEL.value,
                    "api_key_configured": bool(vectordb_config.CUSTOM_EMBEDDING_API_KEY.value)
                }
            }
        }
        
        # 의존성 확인
        try:
            import sentence_transformers
            debug_info["dependencies"]["sentence_transformers"] = {
                "available": True,
                "version": getattr(sentence_transformers, "__version__", "unknown")
            }
        except ImportError as e:
            debug_info["dependencies"]["sentence_transformers"] = {
                "available": False,
                "error": str(e)
            }
        
        try:
            import openai
            debug_info["dependencies"]["openai"] = {
                "available": True,
                "version": getattr(openai, "__version__", "unknown")
            }
        except ImportError as e:
            debug_info["dependencies"]["openai"] = {
                "available": False,
                "error": str(e)
            }
        
        try:
            import aiohttp
            debug_info["dependencies"]["aiohttp"] = {
                "available": True,
                "version": getattr(aiohttp, "__version__", "unknown")
            }
        except ImportError as e:
            debug_info["dependencies"]["aiohttp"] = {
                "available": False,
                "error": str(e)
            }
        
        # 클라이언트 상세 정보
        if rag_service.embeddings_client:
            try:
                client_info = rag_service.embeddings_client.get_provider_info()
                debug_info["client_info"] = client_info
                
                # 기본 사용 가능성 체크
                basic_available = rag_service._check_client_basic_availability(rag_service.embeddings_client)
                debug_info["basic_availability"] = basic_available
                
                # 실제 사용 가능성 체크 (비동기)
                try:
                    full_available = await rag_service.embeddings_client.is_available()
                    debug_info["full_availability"] = full_available
                except Exception as e:
                    debug_info["full_availability"] = False
                    debug_info["availability_error"] = str(e)
                    
            except Exception as e:
                debug_info["client_error"] = str(e)
        
        return debug_info
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get debug info: {str(e)}") 
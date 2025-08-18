"""
Retrieval 컨트롤러

RAG 시스템의 문서 및 컬렉션 관련 API 엔드포인트를 제공합니다.
문서 업로드, 검색, 컬렉션 관리 등의 기능을 담당합니다.
"""

from fastapi import APIRouter, HTTPException, Request, UploadFile, File, Form
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union
import logging
import aiofiles
import aiofiles.os
import os
import json
from pathlib import Path
import datetime
import uuid
from service.database.models.vectordb import VectorDB, VectorDBChunkMeta, VectorDBChunkEdge
from controller.controller_helper import extract_user_id_from_request
from controller.singletonHelper import get_config_composer, get_vector_manager, get_rag_service, get_document_processor, get_db_manager
from service.embedding import get_fastembed_service

logger = logging.getLogger("retrieval-controller")
router = APIRouter(prefix="/api/retrieval", tags=["retrieval"])

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
    collection_make_name: str
    distance: str = "Cosine"
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class DocumentCreateRequest(BaseModel):
    file: UploadFile = File(...)
    collection_name: str = Form(...)
    chunk_size: int = Form(1000)
    chunk_overlap: int = Form(200)
    metadata: Optional[str] = Form(None)

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
    rerank: Optional[bool] = False
    rerank_top_k: Optional[int] = 20

# Collection Management Endpoints
@router.get("/collections")
async def list_collections(request: Request,):
    """모든 컬렉션 목록 조회"""
    user_id = extract_user_id_from_request(request)

    app_db = request.app.state.app_db
    if not app_db:
        raise HTTPException(
            status_code=500,
            detail="Database connection not available"
        )
    try:
        existing_data = app_db.find_by_condition(
            VectorDB,
            {
                "user_id": user_id,
            },
            limit=10000,
            return_list=True
        )
        return existing_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list collections: {str(e)}")

@router.post("/collections")
async def create_collection(request: Request, collection_request: CollectionCreateRequest):
    """새 컬렉션 생성 및 메타 등록"""
    user_id = extract_user_id_from_request(request)

    app_db = request.app.state.app_db
    if not app_db:
        raise HTTPException(
            status_code=500,
            detail="Database connection not available"
        )

    rag_service = get_rag_service(request)
    vector_size = rag_service.config.VECTOR_DIMENSION.value
    vector_manager = rag_service.vector_manager
    # UUID 기반으로 컬렉션 이름 생성
    collection_name = str(uuid.uuid4())
    collection_name = collection_request.collection_make_name+'_'+collection_name

    print(collection_name)

    try:
        # 1. Qdrant에 컬렉션 생성 먼저
        result = vector_manager.create_collection(
            collection_name=collection_name,  # 실제 Qdrant 컬렉션 이름은 UUID
            vector_size=vector_size,
            distance=collection_request.distance,
            description=collection_request.description,
            metadata={
                **(collection_request.metadata or {}),
                "user_id": user_id,
                "original_name": collection_request.collection_make_name,
            }
        )


        if result.get("status") == "created":
            vector_db = VectorDB(
                user_id=user_id,
                collection_make_name=collection_request.collection_make_name,
                collection_name=collection_name,
                description=collection_request.description,
                registered_at=datetime.datetime.now(),
                updated_at=datetime.datetime.now()
            )
            app_db.insert(vector_db)

        return {
            "message": "Collection created",
            "collection_name": collection_name,
            "result": result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create collection: {str(e)}")


@router.delete("/collections")
async def delete_collection(request: Request, collection_request: CollectionDeleteRequest):
    """컬렉션 삭제"""
    try:
        rag_service = get_rag_service(request)
        vector_manager = rag_service.vector_manager

        result = vector_manager.delete_collection(collection_request.collection_name)

        if result.get("status") == "success":
            app_db = request.app.state.app_db
            if not app_db:
                raise HTTPException(
                    status_code=500,
                    detail="Database connection not available"
                )
            rag_service = get_rag_service(request)
            vector_manager = rag_service.vector_manager

            app_db.delete_by_condition(VectorDB, {
                "collection_name": collection_request.collection_name
            })

        return {"message": "Collection deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete collection: {str(e)}")

@router.get("/collections/{collection_name}")
async def get_collection_info(request: Request, collection_name: str):
    """특정 컬렉션 정보 조회"""
    try:
        vector_manager = get_vector_manager(request)
        return vector_manager.get_collection_info(collection_name)
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
    user_id: int = Form(...),
    metadata: Optional[str] = Form(None)
):
    """문서 업로드 및 처리"""
    try:
        rag_service = get_rag_service(request)
        app_db = get_db_manager(request)

        # 파일 저장 (collection명 하위에 저장)
        upload_dir = Path("downloads") / collection_name
        upload_dir.mkdir(parents=True, exist_ok=True)

        file_path = upload_dir / file.filename

        await file.seek(0)
        async with aiofiles.open(file_path, 'wb') as f:
            while True:
                chunk = await file.read(1024 * 1024)  # 1MB
                if not chunk:
                    break
                await f.write(chunk)

        # 메타데이터 파싱
        doc_metadata = {}
        if metadata:
            try:
                doc_metadata = json.loads(metadata)
            except json.JSONDecodeError:
                logger.warning("Invalid metadata JSON: %s", metadata)

        # 문서 처리
        result = await rag_service.process_document(
            user_id,
            app_db,
            str(file_path),
            collection_name,
            chunk_size,
            chunk_overlap,
            doc_metadata,
        )

        # # 임시 파일 삭제
        # try:
        #     await aiofiles.os.remove(file_path)
        # except Exception:
        #     pass

        return result

    except Exception as e:
        logger.error("Document upload failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to upload document: {str(e)}")

@router.post("/documents/search")
async def search_documents(request: Request, search_request: DocumentSearchRequest):
    """문서 검색"""
    try:
        rag_service = get_rag_service(request)
        # support optional rerank parameters
        result = await rag_service.search_documents(
            search_request.collection_name,
            search_request.query_text,
            search_request.limit,
            search_request.score_threshold,
            search_request.filter,
            rerank=search_request.rerank,
            rerank_top_k=search_request.rerank_top_k
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to search documents: {str(e)}")


@router.post("/hybrid/index")
async def hybrid_index_collection(request: Request, source_collection: str = Form(...), target_collection: str = Form(...)):
    """기존 컬렉션의 청크들을 가져와 FastEmbed 기반 하이브리드 컬렉션으로 색인합니다."""
    try:
        rag_service = get_rag_service(request)
        vector_manager = rag_service.vector_manager
        client = vector_manager.client

        # scroll all points from source collection
        points = []
        offset = None
        while True:
            scroll_res = client.scroll(collection_name=source_collection, limit=500, offset=offset, with_payload=True)
            pts, next_offset = scroll_res
            points.extend(pts)
            if next_offset is None:
                break
            offset = next_offset

        if not points:
            return {"message": "No points found in source collection", "count": 0}

        documents = [p.payload.get("chunk_text", "") for p in points]
        payloads = [p.payload for p in points]

        fes = get_fastembed_service()
        fes.init_models()
        dense, sparse, late = fes.embed_documents_all(documents)

        # create hybrid collection and upsert
        dense_key = "all-MiniLM-L6-v2"
        late_key = "colbertv2.0"
        sparse_key = "bm25"

        fes.create_hybrid_collection(client, target_collection, dense_key, len(dense[0]), late_key, len(late[0][0]), sparse_key)
        op = fes.upsert_hybrid_points(client, target_collection, dense_key, sparse_key, late_key, dense, sparse, late, documents, ids=None, payloads=payloads)

        return {"message": "Hybrid index created", "upsert_op": str(op)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create hybrid index: {e}")


@router.post("/hybrid/search")
async def hybrid_search(request: Request, collection_name: str = Form(...), query_text: str = Form(...), limit: int = Form(10)):
    """Hybrd search endpoint using FastEmbedService (prefetch + late rerank)"""
    try:
        rag_service = get_rag_service(request)
        client = rag_service.vector_manager.client
        fes = get_fastembed_service()
        fes.init_models()

        dense_model = fes.dense_model
        sparse_model = fes.sparse_model
        late_model = fes.late_model

        results = fes.hybrid_search(client, collection_name, query_text, dense_model, sparse_model, late_model,
                                     dense_key="all-MiniLM-L6-v2", sparse_key="bm25", late_key="colbertv2.0",
                                     prefetch_limits={"all-MiniLM-L6-v2":20, "bm25":20}, limit=limit)

        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hybrid search failed: {e}")

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

@router.get("/collections/detail/{collection_name}/documents")
async def get_document_detail_meta(request: Request, collection_name: str):
    """특정 문서의 메타데이터 조회"""
    try:
        app_db = get_db_manager(request)
        user_id = extract_user_id_from_request(request)
        existing_data = app_db.find_by_condition(
            VectorDBChunkMeta,
            {
                "user_id": user_id,
                "collection_name": collection_name,
            },
            limit=10000,
            return_list=True
        )
        return existing_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get document details: {str(e)}")

@router.get("/collections-all/detail/documents")
async def get_all_document_detail_meta(request: Request):
    """모든 문서의 메타데이터 조회"""
    try:
        app_db = get_db_manager(request)
        user_id = extract_user_id_from_request(request)
        existing_data = app_db.find_by_condition(
            VectorDBChunkMeta,
            {
                "user_id": user_id,
            },
            limit=10000,
            return_list=True
        )
        return existing_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get document details: {str(e)}")

@router.get("/collections/detail/{collection_name}/edges")
async def get_document_detail_edges(request: Request, collection_name: str):
    """특정 문서의 엣지 메타데이터 조회"""
    try:
        app_db = get_db_manager(request)
        user_id = extract_user_id_from_request(request)
        existing_data = app_db.find_by_condition(
            VectorDBChunkEdge,
            {
                "user_id": user_id,
                "collection_name": collection_name,
            },
            limit=10000,
            return_list=True
        )
        return existing_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get document details: {str(e)}")

@router.get("/collections-all/detail/edges")
async def get_all_document_detail_edges(request: Request):
    """모든 문서의 엣지 메타데이터 조회"""
    try:
        app_db = get_db_manager(request)
        user_id = extract_user_id_from_request(request)
        existing_data = app_db.find_by_condition(
            VectorDBChunkEdge,
            {
                "user_id": user_id,
            },
            limit=10000,
            return_list=True
        )
        return existing_data
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

@router.get("/health")
async def retrieval_health_check(request: Request):
    """retrieval 시스템 상태 확인"""
    try:
        rag_service = get_rag_service(request)

        health_status = {
            "qdrant_client": rag_service.vector_manager.is_connected(),
            "document_processor": bool(rag_service.document_processor),
        }

        if rag_service.vector_manager.is_connected():
            collections = rag_service.vector_manager.list_collections()
            health_status.update({
                "collections_count": len(collections["collections"]),
                "qdrant_status": "connected"
            })
        else:
            health_status["qdrant_status"] = "disconnected"

        overall_status = "healthy" if all([
            rag_service.vector_manager.is_connected(),
            rag_service.document_processor
        ]) else "unhealthy"

        return {
            "status": overall_status,
            "message": "Retrieval system status check",
            "components": health_status
        }
    except Exception as e:
        logger.error("Retrieval health check failed: %s", e)
        return {
            "status": "unhealthy",
            "message": f"Retrieval health check failed: {e}"
        }

@router.get("/debug/info")
async def get_retrieval_debug_info(request: Request):
    """디버깅을 위한 retrieval 상세 정보 조회"""
    try:
        rag_service = get_rag_service(request)
        vectordb_config = rag_service.config

        debug_info = {
            "vector_manager": {
                "connected": rag_service.vector_manager.is_connected(),
                "host": vectordb_config.QDRANT_HOST.value,
                "port": vectordb_config.QDRANT_PORT.value,
                "use_grpc": vectordb_config.QDRANT_USE_GRPC.value,
                "grpc_port": vectordb_config.QDRANT_GRPC_PORT.value,
            },
            "document_processor": {
                "initialized": bool(rag_service.document_processor),
                "supported_types": rag_service.document_processor.get_supported_types() if rag_service.document_processor else []
            },
            "collection_info": {}
        }

        # 컬렉션 정보 추가
        if rag_service.vector_manager.is_connected():
            try:
                collections = rag_service.vector_manager.list_collections()
                debug_info["collection_info"] = collections
            except Exception as e:
                debug_info["collection_info"] = {"error": str(e)}

        return debug_info

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get retrieval debug info: {str(e)}")

@router.get("/config")
async def get_retrieval_config(request: Request):
    """현재 retrieval 시스템 설정 조회"""
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
                }
            }
        else:
            raise HTTPException(status_code=500, detail="Configuration not available")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get retrieval config: {str(e)}")

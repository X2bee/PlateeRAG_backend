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
import os
import json
from pathlib import Path
import datetime
import uuid
from zoneinfo import ZoneInfo
from service.database.models.vectordb import VectorDB, VectorDBFolders, VectorDBChunkMeta, VectorDBChunkEdge
from service.database.models.user import User

from controller.helper.controllerHelper import extract_user_id_from_request
from controller.helper.singletonHelper import get_config_composer, get_vector_manager, get_rag_service, get_document_processor, get_db_manager, get_document_info_generator
from service.embedding import get_fastembed_service

from service.embedding.embedding_factory import EmbeddingFactory
from service.vector_db.vector_manager import VectorManager
from service.retrieval.document_processor.document_processor import DocumentProcessor
from service.retrieval.document_info_generator.document_info_generator import DocumentInfoGenerator

logger = logging.getLogger("retrieval-controller")
router = APIRouter(prefix="/retrieval", tags=["retrieval"])

# 환경변수에서 타임존 가져오기 (기본값: 서울 시간)
TIMEZONE = ZoneInfo(os.getenv('TIMEZONE', 'Asia/Seoul'))

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

class CollectionRemakeRequest(BaseModel):
    collection_name: str
    new_embedding_model: Optional[str] = None

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

@router.get("/collections")
async def list_collections(request: Request):
    """모든 컬렉션 목록 조회"""
    user_id = extract_user_id_from_request(request)
    app_db = get_db_manager(request)
    user = app_db.find_by_id(User, user_id)
    groups = user.groups
    try:
        existing_data = app_db.find_by_condition(
            VectorDB,
            {
                "user_id": user_id,
            },
        limit=10000,
        return_list=True
        )

        if groups and groups != None and groups != [] and len(groups) > 0:
            for group_name in groups:
                shared_data = app_db.find_by_condition(
                    VectorDB,
                    {
                        "share_group": group_name,
                        "is_shared": True,
                    },
                    limit=10000,
                    return_list=True
                )
                existing_data.extend(shared_data)

        seen_ids = set()
        unique_data = []
        for item in existing_data:
            if item['id'] not in seen_ids:
                seen_ids.add(item['id'])
                unique_data.append(item)

        return unique_data
    except Exception as e:
        logger.error(f"Failed to list collections: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list collections: {str(e)}")

@router.post("/update/collections")
async def update_collections(request: Request, update_dict: dict):
    user_id = extract_user_id_from_request(request)
    app_db = get_db_manager(request)

    try:
        existing_data = app_db.find_by_condition(
            VectorDB,
            {
                "user_id": user_id,
                "collection_name": update_dict.get("collection_name")
            },
        )

        if not existing_data:
            raise HTTPException(status_code=404, detail="Collection not found")
        existing_data = existing_data[0]

        existing_data.is_shared = update_dict.get("is_shared", existing_data.is_shared)
        existing_data.share_group = update_dict.get("share_group", existing_data.share_group)

        app_db.update(existing_data)

        logger.info(f"Collection '{existing_data.collection_name}' updated successfully.")
        return {
            "message": "Collection updated successfully",
            "collection_name": existing_data.collection_name
        }

    except Exception as e:
        logger.error(f"Failed to update collection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update collection: {str(e)}")

@router.post("/collections")
async def create_collection(request: Request, collection_request: CollectionCreateRequest):
    """새 컬렉션 생성 및 메타 등록"""
    user_id = extract_user_id_from_request(request)
    app_db = get_db_manager(request)
    config_composer = get_config_composer(request)
    vector_size = config_composer.get_config_by_name("QDRANT_VECTOR_DIMENSION").value
    vector_manager = get_vector_manager(request)

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
                registered_at=datetime.datetime.now(TIMEZONE),
                updated_at=datetime.datetime.now(TIMEZONE),
                vector_size=vector_size,
                is_shared=False,
                share_group=None,
                share_permissions=None,
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
    user_id = extract_user_id_from_request(request)
    app_db = get_db_manager(request)

    existing_collection = app_db.find_by_condition(VectorDB, {'user_id': user_id, 'collection_name': collection_request.collection_name})

    if existing_collection:
        try:
            vector_manager = get_vector_manager(request)
            result = vector_manager.delete_collection(collection_request.collection_name)

            if result.get("status") == "success":
                app_db = get_db_manager(request)
                rag_service = get_rag_service(request)
                vector_manager = rag_service.vector_manager

                app_db.delete_by_condition(VectorDB, {
                    "collection_name": collection_request.collection_name
                })

            return {"message": "Collection deleted"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to delete collection: {str(e)}")

    else:
        raise HTTPException(status_code=404, detail="Collection not found or not owned by user")


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
    metadata: Optional[str] = Form(None),
    process_type: Optional[str] = Form("default")
):
    """문서 업로드 및 처리"""
    try:
        rag_service = get_rag_service(request)
        app_db = get_db_manager(request)
        file_extension = Path(file.filename).suffix[1:].lower() if file.filename else ""

        # 파일 유형에 따른 process_type 검증
        if file_extension == 'pdf':
            valid_process_types = ["default", "text", "ocr"]
            if process_type not in valid_process_types:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid process_type for PDF files. Must be one of: {valid_process_types}"
                )
        elif file_extension in ['docx', 'doc']:
            valid_process_types = ["default", "text", "html", "ocr", "html_pdf_ocr"]
            if process_type not in valid_process_types:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid process_type for DOCX files. Must be one of: {valid_process_types}"
                )
        else:
            # PDF, DOCX가 아닌 파일들은 process_type이 default가 아니면 경고
            if process_type != "default":
                logger.warning(f"process_type '{process_type}' is only supported for PDF and DOCX files. Using 'default' for {file_extension} files.")
                process_type = "default"

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
            use_llm_metadata=True,
            process_type=process_type
        )

        return result

    except Exception as e:
        logger.error("Document upload failed: %s", e)
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
        vector_manager = get_vector_manager(request)
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
        vector_manager = get_vector_manager(request)
        client = vector_manager.client
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
        app_db = get_db_manager(request)
        try:
            directory_info = app_db.find_by_condition(VectorDBFolders, {'collection_name': collection_name})
        except Exception as e:
            directory_info = []
            logger.warning(f"Failed to fetch directory info: {e}")

        rag_service = get_rag_service(request)
        contents = await rag_service.list_documents_in_collection(collection_name)
        contents['directory_info'] = directory_info

        return contents
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")

@router.get("/collections/{collection_name}/documents/{document_id}")
async def get_document_details(request: Request, collection_name: str, document_id: str):
    """특정 문서의 상세 정보 조회"""
    try:
        rag_service = get_rag_service(request)
        rag_default_info = rag_service.get_document_details(collection_name, document_id)

        app_db = get_db_manager(request)
        additional_info = app_db.find_by_condition(VectorDBChunkMeta, {'document_id': document_id, 'collection_name': collection_name})
        if additional_info:
            additional_info = additional_info[0]
            provider_info = {
                "embedding_provider": additional_info.embedding_provider,
                "embedding_model_name": additional_info.embedding_model_name,
                "embedding_dimension": additional_info.embedding_dimension
            }

            rag_default_info.update({
                "provider_info": provider_info
            })

        return rag_default_info
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
        vector_manager = get_vector_manager(request)
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

        return vector_manager.insert_points(
            insert_request.collection_name,
            points_data
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to insert points: {str(e)}")

@router.post("/refresh/rag-system")
async def refresh_retrieval_config(request: Request):
    """현재 retrieval 시스템 설정 조회"""
    try:
        config_composer = get_config_composer(request)
        embedding_client = EmbeddingFactory.create_embedding_client(config_composer)
        request.app.state.embedding_client = embedding_client

        vector_manager = VectorManager(config_composer)
        request.app.state.vector_manager = vector_manager

        document_processor = DocumentProcessor(config_composer)
        request.app.state.document_processor = document_processor

        document_info_generator = DocumentInfoGenerator(config_composer)
        request.app.state.document_info_generator = document_info_generator

        return {
            "message": "Retrieval configuration refreshed successfully"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get retrieval config: {str(e)}")

@router.post("/collections/remake")
async def remake_collection(request: Request, remake_request: CollectionRemakeRequest):
    """컬렉션을 새로운 임베딩 모델로 재생성

    기존 컬렉션의 모든 문서를 보존하면서 새로운 임베딩 차원으로 컬렉션을 재생성합니다.
    """
    user_id = extract_user_id_from_request(request)
    app_db = get_db_manager(request)

    # 사용자 권한 확인
    existing_collection = app_db.find_by_condition(VectorDB, {
        'user_id': user_id,
        'collection_name': remake_request.collection_name
    })

    if not existing_collection:
        raise HTTPException(status_code=404, detail="Collection not found or not owned by user")

    try:
        rag_service = get_rag_service(request)
        result = await rag_service.remake_collection(
            user_id=user_id,
            app_db=app_db,
            collection_name=remake_request.collection_name,
            new_embedding_model=remake_request.new_embedding_model
        )
        return result

    except Exception as e:
        logger.error(f"Failed to remake collection '{remake_request.collection_name}': {e}")
        raise HTTPException(status_code=500, detail=f"Failed to remake collection: {str(e)}")

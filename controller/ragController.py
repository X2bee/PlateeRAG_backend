from fastapi import APIRouter, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union
import logging
from datetime import datetime
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct, Filter, FieldCondition, 
    Range, MatchValue, CollectionInfo, UpdateResult, ScoredPoint
)
import uuid
import asyncio
import aiofiles
import os
import json
from pathlib import Path

# Document processing imports
import PyPDF2
from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.embeddings import EmbeddingFactory

logger = logging.getLogger("vectordb-controller")
router = APIRouter(prefix="/rag", tags=["rag"])

# Enhanced Pydantic Models for RAG
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

# New RAG-specific models
class DocumentUploadRequest(BaseModel):
    collection_name: str
    chunk_size: int = 1000
    chunk_overlap: int = 200
    process_chunks: bool = True
    metadata: Optional[Dict[str, Any]] = None

class DocumentSearchRequest(BaseModel):
    collection_name: str
    query_text: str
    limit: int = 5
    score_threshold: Optional[float] = 0.7
    filter: Optional[Dict[str, Any]] = None

class DocumentSearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    total: int
    query_vector: Optional[List[float]] = None

# Embedding provider models
class EmbeddingProviderSwitchRequest(BaseModel):
    new_provider: str

class EmbeddingTestRequest(BaseModel):
    query_text: str = "Hello, world!"

class RAGController:
    def __init__(self, vectordb_config, openai_config=None):
        """RAG 컨트롤러 초기화"""
        self.config = vectordb_config
        self.openai_config = openai_config  # 하위 호환성을 위해 유지
        self.client = None
        self.embeddings_client = None
        self._initialize_client()
        self._initialize_embeddings()
    
    def _initialize_client(self):
        """Qdrant 클라이언트 초기화"""
        try:
            host = self.config["vectordb"].QDRANT_HOST.value
            port = self.config["vectordb"].QDRANT_PORT.value
            api_key = self.config["vectordb"].QDRANT_API_KEY.value
            use_grpc = self.config["vectordb"].QDRANT_USE_GRPC.value
            grpc_port = self.config["vectordb"].QDRANT_GRPC_PORT.value
            
            if use_grpc:
                self.client = QdrantClient(
                    host=host,
                    grpc_port=grpc_port,
                    api_key=api_key if api_key else None
                )
            else:
                self.client = QdrantClient(
                    host=host,
                    port=port,
                    api_key=api_key if api_key else None
                )
            
            logger.info(f"Qdrant client initialized: {host}:{port}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {e}")
            self.client = None
            raise HTTPException(status_code=500, detail=f"Failed to connect to Qdrant: {e}")
    
    def _initialize_embeddings(self):
        """임베딩 클라이언트 초기화 (팩토리 패턴 사용)"""
        max_retries = 3
        fallback_providers = ["huggingface", "openai", "custom_http"]
        
        for retry in range(max_retries):
            try:
                provider = self.config["vectordb"].EMBEDDING_PROVIDER.value
                logger.info(f"Attempting to initialize embedding client: {provider} (attempt {retry + 1})")
                
                self.embeddings_client = EmbeddingFactory.create_embedding_client(self.config["vectordb"])
                
                # 클라이언트 기본 초기화 체크
                is_available = self._check_client_basic_availability(self.embeddings_client)
                
                if is_available:
                    # 임베딩 차원 수 확인 및 로깅
                    try:
                        dimension = self.embeddings_client.get_embedding_dimension()
                        logger.info(f"Embedding client initialized successfully: {provider}, dimension: {dimension}")
                        
                        # 설정에서 AUTO_DETECT_EMBEDDING_DIM이 True면 차원 수 업데이트
                        if self.config["vectordb"].AUTO_DETECT_EMBEDDING_DIM.value:
                            old_dimension = self.config["vectordb"].VECTOR_DIMENSION.value
                            if old_dimension != dimension:
                                logger.info(f"Updating vector dimension from {old_dimension} to {dimension}")
                                self.config["vectordb"].VECTOR_DIMENSION.value = dimension
                        
                        return
                    except Exception as dim_error:
                        logger.warning(f"Could not get embedding dimension: {dim_error}")
                        logger.info(f"Embedding client initialized successfully: {provider}")
                        return
                else:
                    logger.warning(f"Embedding client created but not available: {provider}")
                    self.embeddings_client = None
                    
            except Exception as e:
                logger.warning(f"Failed to initialize embeddings client '{provider}': {e}")
                self.embeddings_client = None
                
                # 다음 대체 제공자로 시도
                if retry < max_retries - 1:
                    current_provider = self.config["vectordb"].EMBEDDING_PROVIDER.value.lower()
                    
                    # 현재 제공자가 아닌 다른 제공자 찾기
                    for fallback in fallback_providers:
                        if fallback != current_provider:
                            logger.info(f"Trying fallback provider: {fallback}")
                            if self.config["vectordb"].switch_embedding_provider(fallback):
                                break
                    else:
                        # 모든 대체재 시도했지만 실패한 경우 HuggingFace로 강제 설정
                        logger.warning("All fallback providers failed. Forcing HuggingFace provider")
                        self.config["vectordb"].EMBEDDING_PROVIDER.value = "huggingface"
        
        # 모든 시도가 실패한 경우
        if not self.embeddings_client:
            logger.error("Failed to initialize any embedding client after all retries")
    
    def _check_client_basic_availability(self, client) -> bool:
        """클라이언트의 기본 사용 가능성 체크 (동기적)"""
        if not client:
            return False
        
        try:
            # 클라이언트 타입별 기본 체크
            if hasattr(client, 'model'):  # HuggingFace
                return client.model is not None
            elif hasattr(client, 'client'):  # OpenAI
                return client.client is not None
            elif hasattr(client, 'base_url'):  # Custom HTTP
                return bool(client.base_url)
            else:
                return True  # 기타 클라이언트는 기본적으로 사용 가능으로 간주
                
        except Exception as e:
            logger.warning(f"Error checking client availability: {e}")
            return False
    
    async def _ensure_embeddings_client(self):
        """임베딩 클라이언트가 사용 가능한지 확인하고 필요시 재초기화"""
        if not self.embeddings_client:
            logger.info("Embedding client not initialized. Attempting initialization...")
            self._initialize_embeddings()
        
        if self.embeddings_client:
            # 기본 사용 가능성 체크
            basic_check = self._check_client_basic_availability(self.embeddings_client)
            if not basic_check:
                logger.warning("Embedding client basic check failed. Re-initializing...")
                self._initialize_embeddings()
            else:
                try:
                    # 실제 사용 가능성 확인 (네트워크 호출 포함)
                    is_available = await self.embeddings_client.is_available()
                    if not is_available:
                        logger.warning("Embedding client availability check failed. Re-initializing...")
                        self._initialize_embeddings()
                except Exception as e:
                    logger.warning(f"Error checking embedding client availability: {e}")
                    # 네트워크 오류 등은 무시하고 기본 체크가 통과했으면 계속 진행
                    if not self._check_client_basic_availability(self.embeddings_client):
                        self._initialize_embeddings()
        
        if not self.embeddings_client:
            # 사용 가능한 제공자 목록 제공
            available_info = self._get_available_providers_info()
            
            raise HTTPException(
                status_code=500, 
                detail=f"Embeddings client not available. Current provider: {self.config['vectordb'].EMBEDDING_PROVIDER.value}. {available_info}"
            )
    
    def _get_available_providers_info(self) -> str:
        """사용 가능한 제공자 정보 문자열 생성"""
        info_parts = []
        
        # OpenAI 체크
        if self.config["vectordb"].get_openai_api_key():
            info_parts.append("OpenAI (API key configured)")
        
        # HuggingFace 체크
        try:
            import sentence_transformers
            info_parts.append("HuggingFace (sentence-transformers available)")
        except ImportError:
            info_parts.append("HuggingFace (requires: pip install sentence-transformers)")
        
        # Custom HTTP 체크
        custom_url = self.config["vectordb"].CUSTOM_EMBEDDING_URL.value
        if custom_url and custom_url != "http://localhost:8000/v1":
            info_parts.append(f"Custom HTTP ({custom_url})")
        
        if info_parts:
            return f"Available options: {', '.join(info_parts)}"
        else:
            return "Please configure at least one embedding provider."
    
    async def reload_embeddings_client(self):
        """임베딩 클라이언트 강제 재로드"""
        logger.info("Reloading embedding client...")
        self.embeddings_client = None
        self._initialize_embeddings()
        
        if self.embeddings_client:
            is_available = await self.embeddings_client.is_available()
            if is_available:
                logger.info("Embedding client reloaded successfully")
                return True
        
        logger.error("Failed to reload embedding client")
        return False
    
    async def _extract_text_from_file(self, file_path: str, file_type: str) -> str:
        """파일에서 텍스트 추출"""
        try:
            if file_type.lower() == 'pdf':
                return await self._extract_text_from_pdf(file_path)
            elif file_type.lower() in ['docx', 'doc']:
                return await self._extract_text_from_docx(file_path)
            elif file_type.lower() == 'txt':
                return await self._extract_text_from_txt(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
        except Exception as e:
            logger.error(f"Failed to extract text from {file_path}: {e}")
            raise
    
    async def _extract_text_from_pdf(self, file_path: str) -> str:
        """PDF 파일에서 텍스트 추출"""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            logger.error(f"Error extracting text from PDF {file_path}: {e}")
            raise
        return text
    
    async def _extract_text_from_docx(self, file_path: str) -> str:
        """DOCX 파일에서 텍스트 추출"""
        try:
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
        except Exception as e:
            logger.error(f"Error extracting text from DOCX {file_path}: {e}")
            raise
        return text
    
    async def _extract_text_from_txt(self, file_path: str) -> str:
        """TXT 파일에서 텍스트 추출"""
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as file:
                text = await file.read()
        except Exception as e:
            logger.error(f"Error extracting text from TXT {file_path}: {e}")
            raise
        return text
    
    def _chunk_text(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
        """텍스트를 청크로 분할"""
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            chunks = text_splitter.split_text(text)
            logger.info(f"Text split into {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.error(f"Error chunking text: {e}")
            raise
    
    async def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """텍스트 리스트를 임베딩으로 변환"""
        # 입력 검증
        if not texts:
            raise ValueError("Text list cannot be empty")
        
        # 텍스트 정규화 및 필터링
        valid_texts = []
        for i, text in enumerate(texts):
            if text is None:
                logger.warning(f"None text found at index {i}, using placeholder")
                valid_texts.append("empty_text_placeholder")
            else:
                text_str = str(text).strip()
                if not text_str:
                    logger.warning(f"Empty text found at index {i}, using placeholder")
                    text_str = "empty_text_placeholder"
                valid_texts.append(text_str)
        
        # 임베딩 클라이언트 상태 확인 및 자동 복구
        await self._ensure_embeddings_client()
        
        try:
            embeddings = await self.embeddings_client.embed_documents(valid_texts)
            logger.info(f"Generated embeddings for {len(valid_texts)} texts using {self.config['vectordb'].EMBEDDING_PROVIDER.value}")
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            # 한 번 더 재시도 (클라이언트 재초기화 후)
            try:
                logger.info("Retrying embedding generation after client reload...")
                await self.reload_embeddings_client()
                await self._ensure_embeddings_client()
                embeddings = await self.embeddings_client.embed_documents(texts)
                logger.info(f"Embedding generation succeeded on retry")
                return embeddings
            except Exception as retry_error:
                logger.error(f"Embedding generation failed on retry: {retry_error}")
                raise HTTPException(
                    status_code=500, 
                    detail=f"Failed to generate embeddings: {str(e)}. Provider: {self.config['vectordb'].EMBEDDING_PROVIDER.value}"
                )
    
    async def _generate_query_embedding(self, query_text: str) -> List[float]:
        """쿼리 텍스트를 임베딩으로 변환"""
        # 입력 검증
        if not query_text:
            raise ValueError("Query text cannot be None or empty")
        
        # 문자열 정규화
        query_text = str(query_text).strip()
        if not query_text:
            raise ValueError("Query text cannot be empty after normalization")
        
        # 임베딩 클라이언트 상태 확인 및 자동 복구
        await self._ensure_embeddings_client()
        
        try:
            embedding = await self.embeddings_client.embed_query(query_text)
            logger.info(f"Generated query embedding for: {query_text[:50]}... using {self.config['vectordb'].EMBEDDING_PROVIDER.value}")
            return embedding
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            # 한 번 더 재시도 (클라이언트 재초기화 후)
            try:
                logger.info("Retrying query embedding generation after client reload...")
                await self.reload_embeddings_client()
                await self._ensure_embeddings_client()
                embedding = await self.embeddings_client.embed_query(query_text)
                logger.info(f"Query embedding generation succeeded on retry")
                return embedding
            except Exception as retry_error:
                logger.error(f"Query embedding generation failed on retry: {retry_error}")
                raise HTTPException(
                    status_code=500, 
                    detail=f"Failed to generate query embedding: {str(e)}. Provider: {self.config['vectordb'].EMBEDDING_PROVIDER.value}"
                )
    
    def create_collection(self, collection_name: str, vector_size: int, distance: str = "Cosine", description: str = None, metadata: Dict[str, Any] = None):
        """컬렉션 생성 (메타데이터 지원)"""
        if not self.client:
            raise HTTPException(status_code=500, detail="Qdrant client not initialized")
        
        try:
            distance_mapping = {
                "Cosine": Distance.COSINE,
                "Euclidean": Distance.EUCLID,
                "Dot": Distance.DOT
            }
            
            if distance not in distance_mapping:
                raise ValueError(f"Unsupported distance metric: {distance}")
            
            # 컬렉션 생성
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=distance_mapping[distance]
                )
            )
            
            # 메타데이터를 첫 번째 포인트로 저장 (컬렉션 정보)
            if description or metadata:
                collection_metadata = {
                    "type": "collection_metadata",
                    "collection_name": collection_name,
                    "description": description or "",
                    "created_at": datetime.now().isoformat(),
                    "vector_size": vector_size,
                    "distance_metric": distance,
                    "document_count": 0,
                    "custom_metadata": metadata or {}
                }
                
                # 메타데이터용 더미 벡터 생성 (모든 값이 0인 벡터)
                dummy_vector = [0.0] * vector_size
                
                # 메타데이터 포인트용 UUID 생성 (Qdrant 호환)
                metadata_id = str(uuid.uuid4())
                metadata_point = PointStruct(
                    id=metadata_id,
                    vector=dummy_vector,
                    payload=collection_metadata
                )
                
                self.client.upsert(
                    collection_name=collection_name,
                    points=[metadata_point]
                )
            
            logger.info(f"Collection '{collection_name}' created successfully with metadata")
            return {
                "message": f"Collection '{collection_name}' created successfully",
                "collection_id": collection_name,
                "metadata": {
                    "description": description,
                    "custom_metadata": metadata,
                    "vector_size": vector_size,
                    "distance_metric": distance
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to create collection '{collection_name}': {e}")
            raise HTTPException(status_code=500, detail=f"Failed to create collection: {e}")
    
    async def process_document(self, file_path: str, collection_name: str, chunk_size: int = 1000, chunk_overlap: int = 200, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """문서를 처리하여 컬렉션에 저장"""
        if not self.client:
            raise HTTPException(status_code=500, detail="Qdrant client not initialized")
        
        try:
            # 파일 확장자 추출
            file_extension = Path(file_path).suffix[1:].lower()
            
            # 텍스트 추출
            logger.info(f"Extracting text from {file_path}")
            text = await self._extract_text_from_file(file_path, file_extension)
            
            if not text.strip():
                raise ValueError("No text content found in the document")
            
            # 텍스트 청킹
            logger.info(f"Chunking text with size {chunk_size} and overlap {chunk_overlap}")
            chunks = self._chunk_text(text, chunk_size, chunk_overlap)
            
            # 임베딩 생성
            logger.info(f"Generating embeddings for {len(chunks)} chunks")
            embeddings = await self._generate_embeddings(chunks)
            
            # 포인트 생성 및 삽입
            points = []
            document_id = str(uuid.uuid4())
            file_name = Path(file_path).name
            
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_metadata = {
                    "document_id": document_id,
                    "chunk_index": i,
                    "chunk_text": chunk,
                    "file_name": file_name,
                    "file_type": file_extension,
                    "processed_at": datetime.now().isoformat(),
                    "chunk_size": len(chunk),
                    "total_chunks": len(chunks)
                }
                
                # 사용자 정의 메타데이터 추가
                if metadata:
                    chunk_metadata.update(metadata)
                
                # 각 청크마다 새로운 UUID 생성 (Qdrant 호환)
                chunk_id = str(uuid.uuid4())
                
                point = PointStruct(
                    id=chunk_id,
                    vector=embedding,
                    payload=chunk_metadata
                )
                points.append(point)
            
            # 벡터 DB에 삽입
            operation_info = self.client.upsert(
                collection_name=collection_name,
                points=points
            )
            
            # 컬렉션 메타데이터 업데이트 (문서 수 증가)
            await self._update_collection_document_count(collection_name, 1)
            
            logger.info(f"Document processed successfully: {len(chunks)} chunks inserted")
            
            return {
                "message": "Document processed successfully",
                "document_id": document_id,
                "file_name": file_name,
                "chunks_created": len(chunks),
                "operation_id": operation_info.operation_id,
                "status": operation_info.status
            }
            
        except Exception as e:
            logger.error(f"Failed to process document {file_path}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to process document: {e}")
    
    async def search_documents(self, collection_name: str, query_text: str, limit: int = 5, score_threshold: float = 0.7, filter_criteria: Dict[str, Any] = None) -> Dict[str, Any]:
        """문서 검색"""
        if not self.client:
            raise HTTPException(status_code=500, detail="Qdrant client not initialized")
        
        # 입력 검증
        if not query_text or not str(query_text).strip():
            raise HTTPException(status_code=400, detail="Query text cannot be empty")
        
        # 쿼리 텍스트 정규화
        query_text = str(query_text).strip()
        
        try:
            # 쿼리 임베딩 생성
            query_embedding = await self._generate_query_embedding(query_text)
            
            # 필터 설정
            search_filter = None
            if filter_criteria:
                conditions = []
                for key, value in filter_criteria.items():
                    if isinstance(value, dict) and "range" in value:
                        range_filter = value["range"]
                        conditions.append(
                            FieldCondition(
                                key=key,
                                range=Range(
                                    gte=range_filter.get("gte"),
                                    lte=range_filter.get("lte")
                                )
                            )
                        )
                    else:
                        conditions.append(
                            FieldCondition(
                                key=key,
                                match=MatchValue(value=value)
                            )
                        )
                
                # 메타데이터 포인트 제외
                conditions.append(
                    FieldCondition(
                        key="type",
                        match=MatchValue(value="collection_metadata")
                    )
                )
                
                if conditions:
                    search_filter = Filter(must_not=[conditions[-1]], must=conditions[:-1])
            else:
                # 메타데이터 포인트만 제외
                search_filter = Filter(
                    must_not=[
                        FieldCondition(
                            key="type",
                            match=MatchValue(value="collection_metadata")
                        )
                    ]
                )
            
            # 검색 실행
            search_results = self.client.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=search_filter
            )
            
            # 결과 포맷팅
            results = []
            for hit in search_results:
                result = {
                    "id": hit.id,
                    "score": hit.score,
                    "document_id": hit.payload.get("document_id"),
                    "chunk_index": hit.payload.get("chunk_index"),
                    "chunk_text": hit.payload.get("chunk_text"),
                    "file_name": hit.payload.get("file_name"),
                    "file_type": hit.payload.get("file_type"),
                    "metadata": {k: v for k, v in hit.payload.items() if k not in ["chunk_text", "document_id", "chunk_index", "file_name", "file_type"]}
                }
                results.append(result)
            
            logger.info(f"Document search completed: {len(results)} results found")
            
            return {
                "query": query_text,
                "results": results,
                "total": len(results),
                "search_params": {
                    "limit": limit,
                    "score_threshold": score_threshold,
                    "filter": filter_criteria
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to search documents in '{collection_name}': {e}")
            raise HTTPException(status_code=500, detail=f"Failed to search documents: {e}")
    
    async def _update_collection_document_count(self, collection_name: str, increment: int):
        """컬렉션의 문서 수 업데이트"""
        try:
            # 기존 메타데이터 조회 (타입으로 필터링)
            search_results = self.client.scroll(
                collection_name=collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="type",
                            match=MatchValue(value="collection_metadata")
                        ),
                        FieldCondition(
                            key="collection_name",
                            match=MatchValue(value=collection_name)
                        )
                    ]
                ),
                limit=1
            )
            
            if search_results[0]:  # 메타데이터가 존재하는 경우
                metadata_point = search_results[0][0]
                current_count = metadata_point.payload.get("document_count", 0)
                new_count = current_count + increment
                
                # 메타데이터 업데이트를 위한 새로운 포인트 생성
                updated_payload = metadata_point.payload.copy()
                updated_payload["document_count"] = new_count
                updated_payload["last_updated"] = datetime.now().isoformat()
                
                # 메타데이터 포인트의 벡터 확인 및 수정
                if metadata_point.vector is None or len(metadata_point.vector) == 0:
                    # 메타데이터 포인트의 벡터가 없으면 현재 설정된 차원으로 더미 벡터 생성
                    current_dimension = self.config["vectordb"].VECTOR_DIMENSION.value
                    dummy_vector = [0.0] * current_dimension
                else:
                    dummy_vector = metadata_point.vector
                
                updated_point = PointStruct(
                    id=metadata_point.id,
                    vector=dummy_vector,
                    payload=updated_payload
                )
                
                self.client.upsert(
                    collection_name=collection_name,
                    points=[updated_point]
                )
                
        except Exception as e:
            logger.warning(f"Failed to update collection document count: {e}")
            # 실패해도 전체 프로세스는 계속 진행
    
    def delete_collection(self, collection_name: str):
        """컬렉션 삭제"""
        if not self.client:
            raise HTTPException(status_code=500, detail="Qdrant client not initialized")
        
        try:
            self.client.delete_collection(collection_name)
            logger.info(f"Collection '{collection_name}' deleted successfully")
            return {"message": f"Collection '{collection_name}' deleted successfully"}
            
        except Exception as e:
            logger.error(f"Failed to delete collection '{collection_name}': {e}")
            raise HTTPException(status_code=500, detail=f"Failed to delete collection: {e}")
    
    def get_collection_info(self, collection_name: str):
        """컬렉션 정보 조회"""
        if not self.client:
            raise HTTPException(status_code=500, detail="Qdrant client not initialized")
        
        try:
            collection_info = self.client.get_collection(collection_name)
            
            # 기본 정보
            result = {
                "collection_name": collection_name,
                "status": collection_info.status,
                "vectors_count": getattr(collection_info, 'vectors_count', 0),
                "segments_count": getattr(collection_info, 'segments_count', 0),
                "config": {
                    "vector_size": collection_info.config.params.vectors.size,
                    "distance": collection_info.config.params.vectors.distance.name
                }
            }
            
            # 선택적 속성들 (버전에 따라 없을 수 있음)
            if hasattr(collection_info, 'disk_data_size'):
                result["disk_data_size"] = collection_info.disk_data_size
            
            if hasattr(collection_info, 'ram_data_size'):
                result["ram_data_size"] = collection_info.ram_data_size
            
            if hasattr(collection_info, 'indexed_vectors_count'):
                result["indexed_vectors_count"] = collection_info.indexed_vectors_count
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get collection info for '{collection_name}': {e}")
            raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found")
    
    def list_collections(self):
        """모든 컬렉션 목록 조회"""
        if not self.client:
            raise HTTPException(status_code=500, detail="Qdrant client not initialized")
        
        try:
            collections = self.client.get_collections()
            return {
                "collections": [collection.name for collection in collections.collections]
            }
            
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to list collections: {e}")
    
    def insert_points(self, collection_name: str, points: List[VectorPoint]):
        """벡터 포인트 삽입"""
        if not self.client:
            raise HTTPException(status_code=500, detail="Qdrant client not initialized")
        
        try:
            qdrant_points = []
            for point in points:
                # point.id가 없으면 새 UUID 생성, 있으면 유효한 형식인지 확인
                if point.id is None:
                    point_id = str(uuid.uuid4())
                else:
                    # 기존 ID가 정수이거나 유효한 UUID인지 확인
                    try:
                        # 정수로 변환 시도
                        point_id = int(point.id)
                    except (ValueError, TypeError):
                        try:
                            # UUID 형식 검증 시도
                            uuid.UUID(str(point.id))
                            point_id = str(point.id)
                        except ValueError:
                            # 유효하지 않은 형식이면 새 UUID 생성
                            logger.warning(f"Invalid point ID format: {point.id}, generating new UUID")
                            point_id = str(uuid.uuid4())
                
                qdrant_points.append(
                    PointStruct(
                        id=point_id,
                        vector=point.vector,
                        payload=point.payload or {}
                    )
                )
            
            operation_info = self.client.upsert(
                collection_name=collection_name,
                points=qdrant_points
            )
            
            logger.info(f"Inserted {len(qdrant_points)} points into '{collection_name}'")
            return {
                "message": f"Successfully inserted {len(qdrant_points)} points",
                "operation_id": operation_info.operation_id,
                "status": operation_info.status
            }
            
        except Exception as e:
            logger.error(f"Failed to insert points into '{collection_name}': {e}")
            raise HTTPException(status_code=500, detail=f"Failed to insert points: {e}")
    
    def search_points(self, collection_name: str, query: SearchQuery):
        """벡터 유사도 검색"""
        if not self.client:
            raise HTTPException(status_code=500, detail="Qdrant client not initialized")
        
        try:
            search_filter = None
            if query.filter:
                # 간단한 필터 구현 (확장 가능)
                conditions = []
                for key, value in query.filter.items():
                    if isinstance(value, dict) and "range" in value:
                        range_filter = value["range"]
                        conditions.append(
                            FieldCondition(
                                key=key,
                                range=Range(
                                    gte=range_filter.get("gte"),
                                    lte=range_filter.get("lte")
                                )
                            )
                        )
                    else:
                        conditions.append(
                            FieldCondition(
                                key=key,
                                match=MatchValue(value=value)
                            )
                        )
                
                if conditions:
                    search_filter = Filter(must=conditions)
            
            search_results = self.client.search(
                collection_name=collection_name,
                query_vector=query.vector,
                limit=query.limit,
                score_threshold=query.score_threshold,
                query_filter=search_filter
            )
            
            results = []
            for hit in search_results:
                results.append({
                    "id": hit.id,
                    "score": hit.score,
                    "payload": hit.payload,
                    "vector": hit.vector
                })
            
            logger.info(f"Search completed for '{collection_name}': {len(results)} results")
            return {
                "results": results,
                "total": len(results)
            }
            
        except Exception as e:
            logger.error(f"Failed to search in '{collection_name}': {e}")
            raise HTTPException(status_code=500, detail=f"Failed to search: {e}")
    
    def delete_points(self, collection_name: str, point_ids: List[Union[str, int]]):
        """포인트 삭제"""
        if not self.client:
            raise HTTPException(status_code=500, detail="Qdrant client not initialized")
        
        try:
            operation_info = self.client.delete(
                collection_name=collection_name,
                points_selector=point_ids
            )
            
            logger.info(f"Deleted {len(point_ids)} points from '{collection_name}'")
            return {
                "message": f"Successfully deleted {len(point_ids)} points",
                "operation_id": operation_info.operation_id,
                "status": operation_info.status
            }
            
        except Exception as e:
            logger.error(f"Failed to delete points from '{collection_name}': {e}")
            raise HTTPException(status_code=500, detail=f"Failed to delete points: {e}")

    async def list_documents_in_collection(self, collection_name: str) -> Dict[str, Any]:
        """컬렉션 내 모든 문서 목록 조회"""
        if not self.client:
            raise HTTPException(status_code=500, detail="Qdrant client not initialized")
        
        try:
            # 메타데이터 포인트 제외하고 실제 문서 청크들만 조회
            search_filter = Filter(
                must_not=[
                    FieldCondition(
                        key="type",
                        match=MatchValue(value="collection_metadata")
                    )
                ]
            )
            
            # 모든 문서 청크 조회 (페이징 처리)
            all_points = []
            offset = None
            
            while True:
                scroll_result = self.client.scroll(
                    collection_name=collection_name,
                    scroll_filter=search_filter,
                    limit=100,  # 한 번에 100개씩 조회
                    offset=offset,
                    with_payload=True,
                    with_vectors=False  # 벡터는 필요없음
                )
                
                points, next_offset = scroll_result
                all_points.extend(points)
                
                if next_offset is None:
                    break
                offset = next_offset
            
            # document_id별로 그룹핑
            documents = {}
            for point in all_points:
                payload = point.payload
                document_id = payload.get("document_id")
                
                if not document_id:
                    continue
                
                if document_id not in documents:
                    documents[document_id] = {
                        "document_id": document_id,
                        "file_name": payload.get("file_name", "unknown"),
                        "file_type": payload.get("file_type", "unknown"),
                        "processed_at": payload.get("processed_at"),
                        "total_chunks": payload.get("total_chunks", 0),
                        "chunks": [],
                        "metadata": {}
                    }
                    
                    # 사용자 정의 메타데이터 추출 (시스템 필드 제외)
                    system_fields = {
                        "document_id", "chunk_index", "chunk_text", "file_name", 
                        "file_type", "processed_at", "chunk_size", "total_chunks"
                    }
                    for key, value in payload.items():
                        if key not in system_fields:
                            documents[document_id]["metadata"][key] = value
                
                # 청크 정보 추가
                chunk_info = {
                    "chunk_id": str(point.id),
                    "chunk_index": payload.get("chunk_index", 0),
                    "chunk_size": payload.get("chunk_size", 0),
                    "chunk_text_preview": payload.get("chunk_text", "")[:100] + "..." if len(payload.get("chunk_text", "")) > 100 else payload.get("chunk_text", "")
                }
                documents[document_id]["chunks"].append(chunk_info)
            
            # 청크를 chunk_index 순으로 정렬
            for doc in documents.values():
                doc["chunks"].sort(key=lambda x: x["chunk_index"])
                doc["actual_chunks"] = len(doc["chunks"])
            
            # 문서 목록을 업로드 시간 순으로 정렬 (최신순)
            document_list = list(documents.values())
            document_list.sort(
                key=lambda x: x["processed_at"] or "",
                reverse=True
            )
            
            logger.info(f"Found {len(document_list)} documents in collection '{collection_name}'")
            
            return {
                "collection_name": collection_name,
                "total_documents": len(document_list),
                "total_chunks": sum(doc["actual_chunks"] for doc in document_list),
                "documents": document_list
            }
            
        except Exception as e:
            logger.error(f"Failed to list documents in collection '{collection_name}': {e}")
            raise HTTPException(status_code=500, detail=f"Failed to list documents: {e}")

    def get_document_details(self, collection_name: str, document_id: str) -> Dict[str, Any]:
        """특정 문서의 상세 정보 조회"""
        if not self.client:
            raise HTTPException(status_code=500, detail="Qdrant client not initialized")
        
        try:
            # 특정 document_id의 모든 청크 조회
            search_filter = Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id)
                    )
                ]
            )
            
            scroll_result = self.client.scroll(
                collection_name=collection_name,
                scroll_filter=search_filter,
                limit=1000,  # 한 문서의 모든 청크
                with_payload=True,
                with_vectors=False
            )
            
            points, _ = scroll_result
            
            if not points:
                raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found")
            
            # 문서 기본 정보 추출
            first_point = points[0]
            payload = first_point.payload
            
            document_info = {
                "document_id": document_id,
                "file_name": payload.get("file_name", "unknown"),
                "file_type": payload.get("file_type", "unknown"),
                "processed_at": payload.get("processed_at"),
                "total_chunks": len(points),
                "metadata": {}
            }
            
            # 사용자 정의 메타데이터 추출
            system_fields = {
                "document_id", "chunk_index", "chunk_text", "file_name", 
                "file_type", "processed_at", "chunk_size", "total_chunks"
            }
            for key, value in payload.items():
                if key not in system_fields:
                    document_info["metadata"][key] = value
            
            # 모든 청크 정보
            chunks = []
            for point in points:
                chunk_payload = point.payload
                chunk_info = {
                    "chunk_id": str(point.id),
                    "chunk_index": chunk_payload.get("chunk_index", 0),
                    "chunk_size": chunk_payload.get("chunk_size", 0),
                    "chunk_text": chunk_payload.get("chunk_text", "")
                }
                chunks.append(chunk_info)
            
            # 청크를 index 순으로 정렬
            chunks.sort(key=lambda x: x["chunk_index"])
            document_info["chunks"] = chunks
            
            logger.info(f"Retrieved document details for '{document_id}': {len(chunks)} chunks")
            
            return document_info
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to get document details for '{document_id}': {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get document details: {e}")

def get_rag_controller(request: Request):
    """RAG 컨트롤러 의존성 주입"""
    if hasattr(request.app.state, 'config') and request.app.state.config:
        return RAGController(request.app.state.config)
    else:
        raise HTTPException(status_code=500, detail="Configuration not available")

# API Endpoints
@router.get("/health")
async def health_check(request: Request):
    """RAG 시스템 연결 상태 확인"""
    try:
        controller = get_rag_controller(request)
        health_status = {
            "qdrant_client": bool(controller.client),
            "embeddings_client": bool(controller.embeddings_client),
            "embedding_provider": controller.config["vectordb"].EMBEDDING_PROVIDER.value
        }
        
        if controller.client:
            # Qdrant 서버 상태 확인
            collections = controller.client.get_collections()
            health_status.update({
                "collections_count": len(collections.collections),
                "qdrant_status": "connected"
            })
        else:
            health_status["qdrant_status"] = "disconnected"
        
        # 임베딩 클라이언트 상태 상세 확인
        if controller.embeddings_client:
            try:
                is_available = await controller.embeddings_client.is_available()
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
            controller.client, 
            controller.embeddings_client,
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

@router.get("/collections")
async def list_collections(request: Request):
    """모든 컬렉션 목록 조회"""
    try:
        controller = get_rag_controller(request)
        return controller.list_collections()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list collections: {str(e)}")

@router.post("/collections")
async def create_collection(request: Request, collection_request: CollectionCreateRequest):
    """새 컬렉션 생성 (메타데이터 지원)"""
    try:
        controller = get_rag_controller(request)
        return controller.create_collection(
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
        controller = get_rag_controller(request)
        return controller.delete_collection(collection_request.collection_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete collection: {str(e)}")

@router.get("/collections/{collection_name}")
async def get_collection_info(request: Request, collection_name: str):
    """특정 컬렉션 정보 조회"""
    try:
        controller = get_rag_controller(request)
        return controller.get_collection_info(collection_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get collection info: {str(e)}")

# New RAG-specific endpoints
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
        controller = get_rag_controller(request)
        
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
        result = await controller.process_document(
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
        controller = get_rag_controller(request)
        result = await controller.search_documents(
            search_request.collection_name,
            search_request.query_text,
            search_request.limit,
            search_request.score_threshold,
            search_request.filter
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to search documents: {str(e)}")

# Legacy VectorDB endpoints (for backwards compatibility)
@router.post("/points")
async def insert_points(request: Request, insert_request: InsertPointsRequest):
    """벡터 포인트 삽입 (기존 호환성)"""
    try:
        controller = get_rag_controller(request)
        return controller.insert_points(
            insert_request.collection_name,
            insert_request.points
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to insert points: {str(e)}")

@router.post("/search")
async def search_points(request: Request, search_request: SearchRequest):
    """벡터 유사도 검색 (기존 호환성)"""
    try:
        controller = get_rag_controller(request)
        return controller.search_points(
            search_request.collection_name,
            search_request.query
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to search points: {str(e)}")

@router.delete("/points")
async def delete_points(request: Request, delete_request: DeletePointsRequest):
    """포인트 삭제 (기존 호환성)"""
    try:
        controller = get_rag_controller(request)
        return controller.delete_points(
            delete_request.collection_name,
            delete_request.point_ids
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete points: {str(e)}")

@router.get("/collections/{collection_name}/documents")
async def list_documents_in_collection(request: Request, collection_name: str):
    """컬렉션 내 모든 문서 목록 조회"""
    try:
        controller = get_rag_controller(request)
        return await controller.list_documents_in_collection(collection_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")

@router.get("/collections/{collection_name}/documents/{document_id}")
async def get_document_details(request: Request, collection_name: str, document_id: str):
    """특정 문서의 상세 정보 조회"""
    try:
        controller = get_rag_controller(request)
        return controller.get_document_details(collection_name, document_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get document details: {str(e)}")

@router.delete("/collections/{collection_name}/documents/{document_id}")
async def delete_document_from_collection(request: Request, collection_name: str, document_id: str):
    """컬렉션에서 특정 문서 삭제"""
    try:
        controller = get_rag_controller(request)
        
        # 해당 document_id의 모든 포인트 ID 조회
        search_filter = Filter(
            must=[
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=document_id)
                )
            ]
        )
        
        scroll_result = controller.client.scroll(
            collection_name=collection_name,
            scroll_filter=search_filter,
            limit=1000,
            with_payload=False,
            with_vectors=False
        )
        
        points, _ = scroll_result
        
        if not points:
            raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found")
        
        # 모든 관련 포인트 삭제
        point_ids = [str(point.id) for point in points]
        
        operation_info = controller.client.delete(
            collection_name=collection_name,
            points_selector=point_ids
        )
        
        # 컬렉션 문서 수 업데이트
        await controller._update_collection_document_count(collection_name, -1)
        
        logger.info(f"Deleted document '{document_id}' from collection '{collection_name}': {len(point_ids)} chunks removed")
        
        return {
            "message": f"Document '{document_id}' deleted successfully",
            "document_id": document_id,
            "chunks_deleted": len(point_ids),
            "operation_id": operation_info.operation_id,
            "status": operation_info.status
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete document '{document_id}': {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")

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

# 임베딩 관리 API 엔드포인트
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
        controller = get_rag_controller(request)
        results = await EmbeddingFactory.test_all_providers(controller.config["vectordb"])
        return {
            "test_results": results,
            "current_provider": controller.config["vectordb"].EMBEDDING_PROVIDER.value
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to test providers: {str(e)}")

@router.get("/embedding/status")
async def get_embedding_status(request: Request):
    """현재 임베딩 클라이언트 상태 조회"""
    try:
        controller = get_rag_controller(request)
        
        if not controller.embeddings_client:
            return {
                "status": "not_initialized",
                "provider": controller.config["vectordb"].EMBEDDING_PROVIDER.value,
                "available": False
            }
        
        provider_info = controller.embeddings_client.get_provider_info()
        is_available = await controller.embeddings_client.is_available()
        
        return {
            "status": "initialized",
            "provider_info": provider_info,
            "available": is_available
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get embedding status: {str(e)}")

@router.post("/embedding/test-query")
async def test_embedding_query(request: Request, test_request: EmbeddingTestRequest):
    """임베딩 생성 테스트"""
    try:
        controller = get_rag_controller(request)
        
        # 자동 복구 로직이 포함된 임베딩 생성
        embedding = await controller._generate_query_embedding(test_request.query_text)
        
        return {
            "query_text": test_request.query_text,
            "embedding_dimension": len(embedding),
            "embedding_preview": embedding[:5],  # 처음 5개 값만 미리보기
            "provider": controller.config["vectordb"].EMBEDDING_PROVIDER.value,
            "success": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to test embedding: {str(e)}")

@router.post("/embedding/reload")
async def reload_embedding_client(request: Request):
    """임베딩 클라이언트 강제 재로드"""
    try:
        controller = get_rag_controller(request)
        
        success = await controller.reload_embeddings_client()
        
        if success:
            provider_info = controller.embeddings_client.get_provider_info()
            return {
                "success": True,
                "message": "Embedding client reloaded successfully",
                "provider": controller.config["vectordb"].EMBEDDING_PROVIDER.value,
                "provider_info": provider_info
            }
        else:
            return {
                "success": False,
                "message": "Failed to reload embedding client",
                "provider": controller.config["vectordb"].EMBEDDING_PROVIDER.value
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload embedding client: {str(e)}")

@router.post("/embedding/switch-provider")
async def switch_embedding_provider(request: Request, switch_request: EmbeddingProviderSwitchRequest):
    """임베딩 제공자 변경"""
    try:
        controller = get_rag_controller(request)
        new_provider = switch_request.new_provider
        
        # OpenAI 설정과 VectorDB 설정 연결 확인 및 재설정
        if hasattr(request.app.state, 'config') and request.app.state.config:
            app_config = request.app.state.config
            if "vectordb" in app_config and "openai" in app_config:
                app_config["vectordb"].set_openai_config(app_config["openai"])
        
        # 제공자 변경 시도
        old_provider = controller.config["vectordb"].EMBEDDING_PROVIDER.value
        success = controller.config["vectordb"].switch_embedding_provider(new_provider)
        
        if not success:
            # 실패 원인을 더 자세히 분석
            error_msg = f"Cannot switch to provider '{new_provider}'. "
            if new_provider.lower() == "openai":
                openai_key = controller.config["vectordb"].get_openai_api_key()
                if not openai_key:
                    error_msg += "OpenAI API key is not configured."
                else:
                    error_msg += "OpenAI configuration validation failed."
            elif new_provider.lower() == "custom_http":
                custom_url = controller.config["vectordb"].CUSTOM_EMBEDDING_URL.value
                if not custom_url or custom_url == "http://localhost:8000/v1":
                    error_msg += "Custom HTTP URL is not properly configured."
            
            raise HTTPException(status_code=400, detail=error_msg)
        
        # 클라이언트 재로드
        reload_success = await controller.reload_embeddings_client()
        
        if reload_success:
            provider_info = controller.embeddings_client.get_provider_info()
            return {
                "success": True,
                "message": f"Embedding provider switched from {old_provider} to {new_provider}",
                "old_provider": old_provider,
                "new_provider": new_provider,
                "provider_info": provider_info
            }
        else:
            # 실패한 경우 원래 제공자로 롤백
            controller.config["vectordb"].switch_embedding_provider(old_provider)
            await controller.reload_embeddings_client()
            
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
        controller = get_rag_controller(request)
        
        # OpenAI 설정과 VectorDB 설정 연결 확인 및 재설정
        if hasattr(request.app.state, 'config') and request.app.state.config:
            app_config = request.app.state.config
            if "vectordb" in app_config and "openai" in app_config:
                app_config["vectordb"].set_openai_config(app_config["openai"])
        
        old_provider = controller.config["vectordb"].EMBEDDING_PROVIDER.value
        
        # 최적의 제공자로 자동 전환 시도
        switched = controller.config["vectordb"].check_and_switch_to_best_provider()
        
        if switched:
            new_provider = controller.config["vectordb"].EMBEDDING_PROVIDER.value
            
            # 클라이언트 재로드
            reload_success = await controller.reload_embeddings_client()
            
            if reload_success:
                provider_info = controller.embeddings_client.get_provider_info()
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
                controller.config["vectordb"].switch_embedding_provider(old_provider)
                await controller.reload_embeddings_client()
                
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
        controller = get_rag_controller(request)
        config_status = controller.config["vectordb"].get_embedding_provider_status()
        
        # 현재 클라이언트 상태 추가
        if controller.embeddings_client:
            try:
                is_available = await controller.embeddings_client.is_available()
                provider_info = controller.embeddings_client.get_provider_info()
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
        controller = get_rag_controller(request)
        vectordb_config = controller.config["vectordb"]
        
        debug_info = {
            "current_provider": vectordb_config.EMBEDDING_PROVIDER.value,
            "auto_detect_dimension": vectordb_config.AUTO_DETECT_EMBEDDING_DIM.value,
            "vector_dimension": vectordb_config.VECTOR_DIMENSION.value,
            "client_initialized": bool(controller.embeddings_client),
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
        if controller.embeddings_client:
            try:
                client_info = controller.embeddings_client.get_provider_info()
                debug_info["client_info"] = client_info
                
                # 기본 사용 가능성 체크
                basic_available = controller._check_client_basic_availability(controller.embeddings_client)
                debug_info["basic_availability"] = basic_available
                
                # 실제 사용 가능성 체크 (비동기)
                try:
                    full_available = await controller.embeddings_client.is_available()
                    debug_info["full_availability"] = full_available
                except Exception as e:
                    debug_info["full_availability"] = False
                    debug_info["availability_error"] = str(e)
                    
            except Exception as e:
                debug_info["client_error"] = str(e)
        
        return debug_info
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get debug info: {str(e)}") 
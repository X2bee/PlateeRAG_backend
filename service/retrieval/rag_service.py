"""
RAG 서비스 모듈

이 모듈은 RAG(Retrieval-Augmented Generation) 시스템의 핵심 비즈니스 로직을 제공합니다.
문서 처리, 임베딩 생성, 벡터 검색 등의 기능을 조합하여 완전한 RAG 서비스를 구현합니다.
"""

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import HTTPException
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

from service.embedding import EmbeddingFactory
from service.retrieval.document_processor import DocumentProcessor
from service.vector_db.vector_manager import VectorManager

logger = logging.getLogger("rag-service")

class RAGService:
    """RAG 서비스의 핵심 비즈니스 로직을 담당하는 클래스"""

    def __init__(self, vectordb_config, collection_config, openai_config=None):
        """RAGService 초기화

        Args:
            vectordb_config: 벡터 DB 설정 객체
            openai_config: OpenAI 설정 객체 (하위 호환성)
        """
        self.config = vectordb_config
        self.openai_config = openai_config
        self.document_processor = DocumentProcessor(collection_config)
        self.vector_manager = VectorManager(vectordb_config)
        self.embeddings_client = None

        self._initialize_embeddings()

    def _config_refresh(self):
        """설정 갱신 메서드 (필요시 호출)"""
        logger.info("Refreshing RAGService configuration")
        try:
            from main import app
            config_composer = app.state.config_composer
        except ImportError:
            logger.error("Cannot import app from main module")
            raise Exception("Application state not available")

        # 새로운 설정 가져오기
        new_vectordb_config = config_composer.get_config_by_category_name("vectordb")
        new_openai_config = config_composer.get_config_by_category_name("openai")

        self.config = new_vectordb_config
        self.openai_config = new_openai_config

    def _cleanup_vector_manager(self):
        """벡터 매니저 정리"""
        if not self.vector_manager:
            return
        try:
            if hasattr(self, 'vector_manager') and self.vector_manager:
                self.vector_manager.cleanup()
                logger.info("Vector manager cleaned up")
        except Exception as e:
            logger.warning(f"Error cleaning up vector manager: {e}")
        finally:
            self.vector_manager = None
            import gc
            gc.collect()
            logger.info("Embedding client cleanup completed with garbage collection")

    async def _cleanup_embeddings_client(self):
        """임베딩 클라이언트 완전 정리 (메모리 해제)"""
        if not self.embeddings_client:
            return
        try:
            if hasattr(self.embeddings_client, 'cleanup'):
                await self.embeddings_client.cleanup()
                logger.info("Called embedding client cleanup method")
            else:
                logger.warning("Embedding client does not have cleanup method")

        except Exception as e:
            logger.warning("Error during embedding client cleanup: %s", e)
        finally:
            self.embeddings_client = None
            import gc
            gc.collect()
            logger.info("Embedding client cleanup completed with garbage collection")

    def _reinitialize_vector_manager(self):
        """Vector Manager 재초기화 (설정 변경 시 호출)"""
        logger.info("Reinitializing vector manager due to config change")

        # 벡터 DB 설정 로그 출력
        print(f"[DEBUG] Vector DB Config:")
        print(f"  - Host: {self.config.QDRANT_HOST.value}")
        print(f"  - Port: {self.config.QDRANT_PORT.value}")
        print(f"  - Use GRPC: {self.config.QDRANT_USE_GRPC.value}")
        print(f"  - GRPC Port: {self.config.QDRANT_GRPC_PORT.value}")
        print(f"  - Vector Dimension: {self.config.VECTOR_DIMENSION.value}")
        print(f"  - Embedding Provider: {self.config.EMBEDDING_PROVIDER.value}")

        # 기존 vector_manager 정리
        self._config_refresh()
        self._cleanup_vector_manager()

        # 새로운 vector_manager 생성
        try:
            self.vector_manager = VectorManager(self.config)
            logger.info("Vector manager reinitialized successfully")
        except Exception as e:
            logger.error(f"Failed to reinitialize vector manager: {e}")
            self.vector_manager = VectorManager(self.config)

    def _initialize_embeddings(self):
        """임베딩 클라이언트 초기화 (팩토리 패턴 사용)"""
        max_retries = 1
        fallback_providers = ["huggingface", "openai", "custom_http"]

        for retry in range(max_retries):
            try:
                provider = self.config.EMBEDDING_PROVIDER.value
                logger.info(f"Attempting to initialize embedding client: {provider} (attempt {retry + 1})")

                self.embeddings_client = EmbeddingFactory.create_embedding_client(self.config)

                # 클라이언트 기본 초기화 체크
                is_available = self._check_client_basic_availability(self.embeddings_client)

                if is_available:
                    # 임베딩 차원 수 확인 및 로깅
                    try:
                        dimension = self.embeddings_client.get_embedding_dimension()
                        logger.info(f"Embedding client initialized successfully: {provider}, dimension: {dimension}")

                        # 설정에서 AUTO_DETECT_EMBEDDING_DIM이 True면 차원 수 업데이트
                        if self.config.AUTO_DETECT_EMBEDDING_DIM.value:
                            old_dimension = self.config.VECTOR_DIMENSION.value
                            if old_dimension != dimension:
                                logger.info(f"Updating vector dimension from {old_dimension} to {dimension}")
                                self.config.VECTOR_DIMENSION.value = dimension
                                self._reinitialize_vector_manager()

                        return
                    except Exception as dim_error:
                        logger.warning(f"Could not get embedding dimension: {dim_error}")
                        logger.info(f"Embedding client initialized successfully: {provider}")
                        return
                else:
                    logger.warning("Embedding client created but not available: %s", provider)
                    self.embeddings_client = None

            except Exception as e:
                logger.warning("Failed to initialize embeddings client '%s': %s", provider, e)
                self.embeddings_client = None

                # 다음 대체 제공자로 시도
                if retry < max_retries - 1:
                    current_provider = self.config.EMBEDDING_PROVIDER.value.lower()

                    # 현재 제공자가 아닌 다른 제공자 찾기
                    for fallback in fallback_providers:
                        if fallback != current_provider:
                            logger.info(f"Trying fallback provider: {fallback}")
                            if self.config.switch_embedding_provider(fallback):
                                self._reinitialize_vector_manager()
                                break
                    else:
                        # 모든 대체재 시도했지만 실패한 경우 HuggingFace로 강제 설정
                        logger.warning("All fallback providers failed. Forcing HuggingFace provider")
                        old_provider = self.config.EMBEDDING_PROVIDER.value
                        self.config.EMBEDDING_PROVIDER.value = "huggingface"
                        if old_provider != "huggingface":
                            # provider가 변경되었으므로 vector_manager 재초기화
                            self._reinitialize_vector_manager()

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

    async def ensure_embeddings_client(self):
        """임베딩 클라이언트가 사용 가능한지 확인하고 필요시 재초기화"""
        if not self.embeddings_client:
            logger.info("Embedding client not initialized. Attempting initialization...")
            self._initialize_embeddings()

        if self.embeddings_client:
            # 기본 사용 가능성 체크
            basic_check = self._check_client_basic_availability(self.embeddings_client)
            if not basic_check:
                logger.warning("Embedding client basic check failed. Re-initializing...")
                await self._cleanup_embeddings_client()
                self._initialize_embeddings()
            else:
                try:
                    # 실제 사용 가능성 확인 (네트워크 호출 포함)
                    is_available = await self.embeddings_client.is_available()
                    if not is_available:
                        logger.warning("Embedding client availability check failed. Re-initializing...")
                        await self._cleanup_embeddings_client()
                        self._initialize_embeddings()
                except Exception as e:
                    logger.warning(f"Error checking embedding client availability: {e}")
                    # 네트워크 오류 등은 무시하고 기본 체크가 통과했으면 계속 진행
                    if not self._check_client_basic_availability(self.embeddings_client):
                        await self._cleanup_embeddings_client()
                        self._initialize_embeddings()

        if not self.embeddings_client:
            # 사용 가능한 제공자 목록 제공
            available_info = self._get_available_providers_info()

            raise HTTPException(
                status_code=500,
                detail=f"Embeddings client not available. Current provider: {self.config.EMBEDDING_PROVIDER.value}. {available_info}"
            )

    def _get_available_providers_info(self) -> str:
        """사용 가능한 제공자 정보 문자열 생성"""
        info_parts = []

        # OpenAI 체크
        if self.config.get_openai_api_key():
            info_parts.append("OpenAI (API key configured)")

        # HuggingFace 체크
        try:
            import sentence_transformers
            info_parts.append("HuggingFace (sentence-transformers available)")
        except ImportError:
            info_parts.append("HuggingFace (requires: pip install sentence-transformers)")

        # Custom HTTP 체크
        custom_url = self.config.CUSTOM_EMBEDDING_URL.value
        if custom_url and custom_url != "http://localhost:8000/v1":
            info_parts.append(f"Custom HTTP ({custom_url})")

        if info_parts:
            return f"Available options: {', '.join(info_parts)}"
        else:
            return "Please configure at least one embedding provider."

    async def reload_embeddings_client(self):
        """임베딩 클라이언트 강제 재로드"""

        logger.info("Reloading embedding client...")
        self._reinitialize_vector_manager()
        await self._cleanup_embeddings_client()

        self._initialize_embeddings()

        if self.embeddings_client:
            is_available = await self.embeddings_client.is_available()
            if is_available:
                logger.info("Embedding client reloaded successfully")
                return True

        logger.error("Failed to reload embedding client")
        return False

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """텍스트 리스트를 임베딩으로 변환

        Args:
            texts: 임베딩으로 변환할 텍스트 리스트

        Returns:
            임베딩 벡터 리스트

        Raises:
            HTTPException: 임베딩 생성 실패
        """
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
        await self.ensure_embeddings_client()

        try:
            embeddings = await self.embeddings_client.embed_documents(valid_texts)
            logger.info(f"Generated embeddings for {len(valid_texts)} texts using {self.config.EMBEDDING_PROVIDER.value}")
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            # 한 번 더 재시도 (클라이언트 재초기화 후)
            try:
                logger.info("Retrying embedding generation after client reload...")
                await self.reload_embeddings_client()
                await self.ensure_embeddings_client()
                embeddings = await self.embeddings_client.embed_documents(texts)
                logger.info(f"Embedding generation succeeded on retry")
                return embeddings
            except Exception as retry_error:
                logger.error(f"Embedding generation failed on retry: {retry_error}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to generate embeddings: {str(e)}. Provider: {self.config.EMBEDDING_PROVIDER.value}"
                )

    async def generate_query_embedding(self, query_text: str, debug: bool = False) -> List[float]:
        """쿼리 텍스트를 임베딩으로 변환

        Args:
            query_text: 쿼리 텍스트
            debug: 디버그 모드 (True시 디버그 로그 출력)

        Returns:
            쿼리 임베딩 벡터 (List[float])

        Raises:
            HTTPException: 임베딩 생성 실패
        """
        # 입력 검증
        if not query_text:
            raise ValueError("Query text cannot be None or empty")

        # 문자열 정규화
        query_text = str(query_text).strip()
        if not query_text:
            raise ValueError("Query text cannot be empty after normalization")

        # 임베딩 클라이언트 상태 확인 및 자동 복구
        await self.ensure_embeddings_client()

        try:
            embedding = await self.embeddings_client.embed_query(query_text)

            if debug:
                vector_stats = {
                    "min": float(min(embedding)),
                    "max": float(max(embedding)),
                    "mean": float(sum(embedding) / len(embedding)),
                    "norm": float(sum(x**2 for x in embedding)**0.5)
                }

                logger.info("[DEBUG] Generated query embedding for: '%s...' using %s", query_text[:100], self.config.EMBEDDING_PROVIDER.value)
                logger.info("[DEBUG] Embedding dimension: %s", len(embedding))
                logger.info("[DEBUG] Vector stats: %s", vector_stats)
                logger.info("[DEBUG] Full embedding vector: %s", embedding)
            else:
                logger.info("Generated query embedding for: %s... using %s", query_text[:50], self.config.EMBEDDING_PROVIDER.value)

            return embedding

        except Exception as e:
            logger.error("Error generating query embedding: %s", e)
            # 한 번 더 재시도 (클라이언트 재초기화 후)
            try:
                logger.info("Retrying query embedding generation after client reload...")
                await self.reload_embeddings_client()
                await self.ensure_embeddings_client()
                embedding = await self.embeddings_client.embed_query(query_text)

                if debug:
                    vector_stats = {
                        "min": float(min(embedding)),
                        "max": float(max(embedding)),
                        "mean": float(sum(embedding) / len(embedding)),
                        "norm": float(sum(x**2 for x in embedding)**0.5)
                    }

                    logger.info("[DEBUG] Query embedding generation succeeded on retry")
                    logger.info("[DEBUG] Embedding dimension: %s", len(embedding))
                    logger.info("[DEBUG] Vector stats: %s", vector_stats)
                    logger.info("[DEBUG] Full embedding vector: %s", embedding)
                else:
                    logger.info("Query embedding generation succeeded on retry")

                return embedding

            except Exception as retry_error:
                logger.error("Query embedding generation failed on retry: %s", retry_error)
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to generate query embedding: {str(e)}. Provider: {self.config.EMBEDDING_PROVIDER.value}"
                ) from retry_error

    async def process_document(self, file_path: str, collection_name: str,
                             chunk_size: int = 1000, chunk_overlap: int = 200,
                             metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """문서를 처리하여 컬렉션에 저장

        Args:
            file_path: 처리할 문서 파일 경로
            collection_name: 저장할 컬렉션 이름
            chunk_size: 청크 크기
            chunk_overlap: 청크 간 중복 크기
            metadata: 사용자 정의 메타데이터

        Returns:
            처리 결과 정보

        Raises:
            HTTPException: 문서 처리 실패
        """
        if not self.vector_manager.is_connected():
            raise HTTPException(status_code=500, detail="Vector database not connected")

        try:
            # 파일 확장자 추출 및 검증
            is_valid, file_extension = self.document_processor.validate_file_format(file_path)
            if not is_valid:
                raise ValueError(f"Unsupported file type: {file_extension}")

            # 텍스트 추출
            logger.info(f"Extracting text from {file_path}")
            text = await self.document_processor.extract_text_from_file(file_path, file_extension)

            if not text.strip():
                raise ValueError("No text content found in the document")

            # 텍스트 청킹
            logger.info(f"Chunking text with size {chunk_size} and overlap {chunk_overlap}")
            chunks = self.document_processor.chunk_text(text, chunk_size, chunk_overlap)

            # 임베딩 생성
            logger.info(f"Generating embeddings for {len(chunks)} chunks")
            embeddings = await self.generate_embeddings(chunks)

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

                # 각 청크마다 새로운 UUID 생성
                chunk_id = str(uuid.uuid4())

                point = {
                    "id": chunk_id,
                    "vector": embedding,
                    "payload": chunk_metadata
                }
                points.append(point)

            # 벡터 DB에 삽입
            operation_info = self.vector_manager.insert_points(collection_name, points)

            # 컬렉션 메타데이터 업데이트 (문서 수 증가)
            self.vector_manager.update_collection_document_count(collection_name, 1)

            logger.info(f"Document processed successfully: {len(chunks)} chunks inserted")

            return {
                "message": "Document processed successfully",
                "document_id": document_id,
                "file_name": file_name,
                "chunks_created": len(chunks),
                "operation_id": operation_info.get("operation_id"),
                "status": operation_info.get("status")
            }

        except Exception as e:
            logger.error(f"Failed to process document {file_path}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to process document: {e}")

    async def search_documents(self, collection_name: str, query_text: str,
                             limit: int = 5, score_threshold: float = 0.7,
                             filter_criteria: Dict[str, Any] = None) -> Dict[str, Any]:
        """문서 검색

        Args:
            collection_name: 검색할 컬렉션 이름
            query_text: 검색 쿼리 텍스트
            limit: 반환할 최대 결과 수
            score_threshold: 최소 유사도 임계값
            filter_criteria: 필터 조건

        Returns:
            검색 결과

        Raises:
            HTTPException: 검색 실패
        """
        if not self.vector_manager.is_connected():
            raise HTTPException(status_code=500, detail="Vector database not connected")

        # 입력 검증
        if not query_text or not str(query_text).strip():
            raise HTTPException(status_code=400, detail="Query text cannot be empty")

        # 쿼리 텍스트 정규화
        query_text = str(query_text).strip()

        try:
            # 쿼리 임베딩 생성
            query_embedding = await self.generate_query_embedding(query_text)

            # 필터 설정 (메타데이터 포인트 제외)
            search_filter = None
            if filter_criteria:
                # 기존 필터 조건들을 must 조건으로 추가
                must_conditions = []
                for key, value in filter_criteria.items():
                    if isinstance(value, dict) and "range" in value:
                        range_filter = value["range"]
                        must_conditions.append(
                            FieldCondition(
                                key=key,
                                range=Range(
                                    gte=range_filter.get("gte"),
                                    lte=range_filter.get("lte")
                                )
                            )
                        )
                    else:
                        must_conditions.append(
                            FieldCondition(
                                key=key,
                                match=MatchValue(value=value)
                            )
                        )

                # 메타데이터 포인트 제외 조건을 must_not에 추가
                search_filter = Filter(
                    must=must_conditions,
                    must_not=[
                        FieldCondition(
                            key="type",
                            match=MatchValue(value="collection_metadata")
                        )
                    ]
                )
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

            # 검색 실행 (vector_manager의 search_points 대신 직접 클라이언트 사용)
            search_results = self.vector_manager.client.search(
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
                    "metadata": {k: v for k, v in hit.payload.items()
                               if k not in ["chunk_text", "document_id", "chunk_index", "file_name", "file_type"]}
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

    async def list_documents_in_collection(self, collection_name: str) -> Dict[str, Any]:
        """컬렉션 내 모든 문서 목록 조회

        Args:
            collection_name: 조회할 컬렉션 이름

        Returns:
            문서 목록 정보

        Raises:
            HTTPException: 조회 실패
        """
        if not self.vector_manager.is_connected():
            raise HTTPException(status_code=500, detail="Vector database not connected")

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
                scroll_result = self.vector_manager.client.scroll(
                    collection_name=collection_name,
                    scroll_filter=search_filter,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
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
        """특정 문서의 상세 정보 조회

        Args:
            collection_name: 컬렉션 이름
            document_id: 문서 ID

        Returns:
            문서 상세 정보

        Raises:
            HTTPException: 조회 실패
        """
        if not self.vector_manager.is_connected():
            raise HTTPException(status_code=500, detail="Vector database not connected")

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

            scroll_result = self.vector_manager.client.scroll(
                collection_name=collection_name,
                scroll_filter=search_filter,
                limit=1000,
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

    def delete_document_from_collection(self, collection_name: str, document_id: str) -> Dict[str, Any]:
        """컬렉션에서 특정 문서 삭제

        Args:
            collection_name: 컬렉션 이름
            document_id: 삭제할 문서 ID

        Returns:
            삭제 결과 정보

        Raises:
            HTTPException: 삭제 실패
        """
        if not self.vector_manager.is_connected():
            raise HTTPException(status_code=500, detail="Vector database not connected")

        try:
            # 해당 document_id의 모든 포인트 ID 조회
            search_filter = Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id)
                    )
                ]
            )

            scroll_result = self.vector_manager.client.scroll(
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

            operation_info = self.vector_manager.delete_points(collection_name, point_ids)

            # 컬렉션 문서 수 업데이트
            self.vector_manager.update_collection_document_count(collection_name, -1)

            logger.info(f"Deleted document '{document_id}' from collection '{collection_name}': {len(point_ids)} chunks removed")

            return {
                "message": f"Document '{document_id}' deleted successfully",
                "document_id": document_id,
                "chunks_deleted": len(point_ids),
                "operation_id": operation_info.get("operation_id"),
                "status": operation_info.get("status")
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to delete document '{document_id}': {e}")
            raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")

    def get_embedding_status(self) -> Dict[str, Any]:
        """현재 임베딩 클라이언트 상태 조회

        Returns:
            임베딩 클라이언트 상태 정보
        """
        if not self.embeddings_client:
            return {
                "status": "not_initialized",
                "provider": self.config.EMBEDDING_PROVIDER.value,
                "available": False
            }

        try:
            provider_info = self.embeddings_client.get_provider_info()
            return {
                "status": "initialized",
                "provider_info": provider_info,
                "available": True  # 기본적으로 true, 실제 확인은 비동기로 해야 함
            }
        except Exception as e:
            return {
                "status": "error",
                "provider": self.config.EMBEDDING_PROVIDER.value,
                "error": str(e),
                "available": False
            }

    def get_config(self) -> Dict[str, Any]:
        """현재 RAG 서비스의 모든 설정 반환

        Returns:
            RAG 서비스 설정 정보
        """
        try:
            config_info = {
                "vector_db": {
                    "host": self.config.QDRANT_HOST.value,
                    "port": self.config.QDRANT_PORT.value,
                    "use_grpc": self.config.QDRANT_USE_GRPC.value,
                    "grpc_port": self.config.QDRANT_GRPC_PORT.value,
                    "collection_name": self.config.COLLECTION_NAME.value,
                    "vector_dimension": self.config.VECTOR_DIMENSION.value,
                    "replicas": self.config.REPLICAS.value,
                    "shards": self.config.SHARDS.value,
                    "connected": self.vector_manager.is_connected() if self.vector_manager else False
                },
                "embedding": {
                    "provider": self.config.EMBEDDING_PROVIDER.value,
                    "auto_detect_dimension": self.config.AUTO_DETECT_EMBEDDING_DIM.value,
                    "client_initialized": bool(self.embeddings_client),
                    "openai": {
                        "api_key_configured": bool(self.config.get_openai_api_key()),
                        "model": self.config.OPENAI_EMBEDDING_MODEL.value,
                        "api_key_length": len(self.config.get_openai_api_key()) if self.config.get_openai_api_key() else 0
                    },
                    "huggingface": {
                        "model_name": self.config.HUGGINGFACE_MODEL_NAME.value,
                        "api_key_configured": bool(self.config.HUGGINGFACE_API_KEY.value),
                        "api_key_length": len(self.config.HUGGINGFACE_API_KEY.value) if self.config.HUGGINGFACE_API_KEY.value else 0
                    },
                    "custom_http": {
                        "url": self.config.CUSTOM_EMBEDDING_URL.value,
                        "model": self.config.CUSTOM_EMBEDDING_MODEL.value,
                        "api_key_configured": bool(self.config.CUSTOM_EMBEDDING_API_KEY.value),
                        "api_key_length": len(self.config.CUSTOM_EMBEDDING_API_KEY.value) if self.config.CUSTOM_EMBEDDING_API_KEY.value else 0
                    }
                },
                "document_processor": {
                    "initialized": bool(self.document_processor),
                    "supported_types": self.document_processor.get_supported_types() if self.document_processor else []
                },
                "service_status": {
                    "initialized": True,
                    "components": {
                        "vector_manager": bool(self.vector_manager),
                        "embeddings_client": bool(self.embeddings_client),
                        "document_processor": bool(self.document_processor)
                    }
                }
            }

            # 임베딩 클라이언트 상세 정보 추가
            if self.embeddings_client:
                try:
                    provider_info = self.embeddings_client.get_provider_info()
                    config_info["embedding"]["provider_info"] = provider_info

                    # 차원 정보 추가
                    try:
                        dimension = self.embeddings_client.get_embedding_dimension()
                        config_info["embedding"]["actual_dimension"] = dimension
                    except Exception:
                        config_info["embedding"]["actual_dimension"] = "unknown"

                except Exception as e:
                    config_info["embedding"]["provider_info_error"] = str(e)

            return config_info

        except Exception as e:
            logger.error(f"Failed to get RAG service config: {e}")
            return {
                "error": str(e),
                "basic_info": {
                    "provider": self.config.EMBEDDING_PROVIDER.value if self.config else "unknown",
                    "vector_db_connected": self.vector_manager.is_connected() if self.vector_manager else False,
                    "embedding_client_initialized": bool(self.embeddings_client)
                }
            }

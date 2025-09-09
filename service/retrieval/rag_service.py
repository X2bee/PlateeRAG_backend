"""
RAG 서비스 모듈

이 모듈은 RAG(Retrieval-Augmented Generation) 시스템의 핵심 비즈니스 로직을 제공합니다.
문서 처리, 임베딩 생성, 벡터 검색 등의 기능을 조합하여 완전한 RAG 서비스를 구현합니다.
"""
import re
import logging
import uuid

import os
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import List, Dict, Any
from fastapi import HTTPException
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range
from service.database.models.vectordb import VectorDB, VectorDBChunkEdge, VectorDBChunkMeta

logger = logging.getLogger("rag-service")

# 환경변수에서 타임존 가져오기 (기본값: 서울 시간)
TIMEZONE = ZoneInfo(os.getenv('TIMEZONE', 'Asia/Seoul'))

class RAGService:
    def __init__(self, config_composer, embedding_client=None, vector_manager=None, document_processor=None, document_info_generator=None):
        logger.info("Initializing RAGService components...")

        self.config_composer = config_composer
        self.embeddings_client = embedding_client
        self.vector_manager = vector_manager

        self.document_processor = document_processor
        self.document_processor.test()

        self.metadata_generator = document_info_generator

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

        try:
            embeddings = await self.embeddings_client.embed_documents(valid_texts)
            logger.info(f"Generated embeddings for {len(valid_texts)} texts using {self.config_composer.get_config_by_name('EMBEDDING_PROVIDER').value}")
            return embeddings
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate embeddings: {str(e)}. Provider: {self.config_composer.get_config_by_name('EMBEDDING_PROVIDER').value}"
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

        try:
            embedding = await self.embeddings_client.embed_query(query_text)

            if debug:
                vector_stats = {
                    "min": float(min(embedding)),
                    "max": float(max(embedding)),
                    "mean": float(sum(embedding) / len(embedding)),
                    "norm": float(sum(x**2 for x in embedding)**0.5)
                }

                logger.info("[DEBUG] Generated query embedding for: '%s...' using %s", query_text[:100], self.config_composer.get_config_by_name('EMBEDDING_PROVIDER').value)
                logger.info("[DEBUG] Embedding dimension: %s", len(embedding))
                logger.info("[DEBUG] Vector stats: %s", vector_stats)
                logger.info("[DEBUG] Full embedding vector: %s", embedding)
            else:
                logger.info("Generated query embedding for: %s... using %s", query_text[:50], self.config_composer.get_config_by_name('EMBEDDING_PROVIDER').value)

            return embedding

        except Exception as e:
            logger.error("Error generating query embedding: %s", e)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate query embedding: {str(e)}. Provider: {self.config_composer.get_config_by_name('EMBEDDING_PROVIDER').value}"
            ) from e

    async def process_document(self, user_id, app_db, file_path: str, collection_name: str,
                                chunk_size: int = 1000, chunk_overlap: int = 200,
                                metadata: Dict[str, Any] = None,
                                use_llm_metadata: bool = True,
                                process_type: str = "default"
                                ) -> Dict[str, Any]:
        """문서를 처리하여 컬렉션에 저장

        Args:
            file_path: 처리할 문서 파일 경로
            collection_name: 저장할 컬렉션 이름
            chunk_size: 청크 크기
            chunk_overlap: 청크 간 중복 크기
            metadata: 사용자 정의 메타데이터
            use_llm_metadata: LLM을 사용한 메타데이터 생성 여부

        Returns:
            처리 결과 정보

        Raises:
            HTTPException: 문서 처리 실패
        """
        if not self.vector_manager.is_connected():
            raise HTTPException(status_code=500, detail="Vector database not connected")

        try:
            processed_chunks = []
            is_valid, file_extension = self.document_processor.validate_file_format(file_path)
            if not is_valid:
                raise ValueError(f"Unsupported file type: {file_extension}")

            # 텍스트 추출
            logger.info(f"Extracting text from {file_path}")
            text = await self.document_processor.extract_text_from_file(file_path, file_extension, process_type)

            if not text.strip():
                raise ValueError("No text content found in the document")

            logger.info(f"Chunking text with size {chunk_size} and overlap {chunk_overlap}")
            try:
                chunks_with_metadata = self.document_processor.chunk_text_with_metadata(
                    text, file_extension, chunk_size, chunk_overlap
                )
            except AttributeError:
                chunks = self.document_processor.chunk_text(text, chunk_size, chunk_overlap)

                if hasattr(self.document_processor, '_build_line_starts'):
                    line_starts = self.document_processor._build_line_starts(text)
                    pos_to_line = getattr(self.document_processor, '_pos_to_line')
                    find_pos = getattr(self.document_processor, '_find_chunk_position')
                    get_page = getattr(self.document_processor, '_get_page_number_for_chunk', None)
                else:
                    # 최소한의 대체 구현
                    line_starts = [0]
                    def pos_to_line(p, ls=line_starts):
                        return 1
                    def find_pos(chunk, full_text, sp=0):
                        return full_text.find(chunk, sp)
                    find_pos = find_pos
                    get_page = None

                chunks_with_metadata = []
                search_pos = 0
                last_line_end = 0
                for i, chunk in enumerate(chunks):
                    chunk_pos = find_pos(chunk, text, search_pos)
                    if chunk_pos != -1:
                        start_pos = chunk_pos
                        end_pos = min(chunk_pos + len(chunk) - 1, len(text) - 1)
                        try:
                            line_start = pos_to_line(start_pos, line_starts)
                            line_end = pos_to_line(end_pos, line_starts)
                        except Exception:
                            # fallback simple estimate
                            line_start = text[:start_pos].count('\n') + 1
                            line_end = text[:end_pos].count('\n') + 1
                        search_pos = chunk_pos + len(chunk)
                        last_line_end = line_end
                    else:
                        lines_in_chunk = chunk.count('\n') or 1
                        line_start = last_line_end + 1
                        line_end = line_start + lines_in_chunk - 1
                        last_line_end = line_end

                    if get_page is not None:
                        try:
                            page_number = get_page(chunk, chunk_pos, [])
                        except Exception:
                            page_number = 1
                    else:
                        page_number = 1

                    chunks_with_metadata.append({
                        "text": chunk,
                        "page_number": page_number,
                        "line_start": line_start,
                        "line_end": line_end,
                        "chunk_index": i
                    })

            chunks = [chunk_data["text"] for chunk_data in chunks_with_metadata]

            #TODO LLM으로 chunk 데이터 Gen하는 것 임시로 구현. 후에 검토 필요
            llm_gen_chunk_metadatas = None
            if use_llm_metadata and await self.metadata_generator.is_enabled():
                logger.info(f"Generating LLM metadata for {len(chunks)} chunks")

                try:
                    llm_gen_chunk_metadatas = await self.metadata_generator.generate_batch_metadata(
                        chunks, None, max_concurrent=3
                    )
                except Exception as e:
                    llm_gen_chunk_metadatas = None

            processed_chunks = []
            if llm_gen_chunk_metadatas:
                for i, (chunk, chunk_metadata) in enumerate(zip(chunks, llm_gen_chunk_metadatas)):
                    summary = chunk_metadata.get("summary")
                    summary_info = f"문서 요약: {summary}" if summary and summary.strip() else ""
                    additional_info = {
                        "keywords": chunk_metadata.get("keywords", []),
                        "topics": chunk_metadata.get("topics", []),
                        "entities": chunk_metadata.get("entities", []),
                        "sentiment": chunk_metadata.get("sentiment", ""),
                        "document_type": chunk_metadata.get("document_type", ""),
                        "complexity_level": chunk_metadata.get("complexity_level", ""),
                        "main_concepts": chunk_metadata.get("main_concepts", [])
                    }

                    uuid_pattern = r'_[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
                    display_collection_name = re.sub(uuid_pattern, '', collection_name, flags=re.IGNORECASE)
                    processed_chunk = f"""이는 '{display_collection_name}' 콜렉션에 존재하는 {(Path(file_path).name)} 파일의 내용입니다.
{summary_info}
{additional_info}

{chunk}"""
                    processed_chunks.append(processed_chunk)

            logger.info(f"Generating embeddings for {len(chunks)} chunks")
            if processed_chunks and len(processed_chunks) == len(chunks):
                logger.info("Using processed chunks for embedding generation")
                embeddings = await self.generate_embeddings(processed_chunks)
                # processed_chunks를 사용하여 청크 업데이트
                chunks = processed_chunks
            else:
                embeddings = await self.generate_embeddings(chunks)

            # 포인트 생성 및 삽입
            points = []
            document_id = str(uuid.uuid4())
            file_name = Path(file_path).name
            # 저장된 파일을 컬렉션 기준으로 식별할 수 있게 경로 형태로 보관
            stored_file_path = f"{collection_name}/{file_name}"
            # 청크별 메타데이터에 병합할 기본 메타데이터 초기화
            if metadata and isinstance(metadata, dict):
                final_metadata = dict(metadata)  # copy incoming metadata
            else:
                final_metadata = {}

            # 공통 메타데이터 보강
            final_metadata.update({
                "user_id": user_id,
                "collection_name": collection_name,
                "file_name": file_name,
                "file_path": stored_file_path
            })

            # llm_gen_chunk_metadatas가 None인 경우 빈 딕셔너리 리스트로 대체
            if llm_gen_chunk_metadatas is None:
                llm_gen_chunk_metadatas = [{} for _ in range(len(chunks_with_metadata))]

            for i, (chunk, chunk_data, embedding, llm_gen_chunk_metadata) in enumerate(zip(chunks, chunks_with_metadata, embeddings, llm_gen_chunk_metadatas)):
                # document_processor에서 계산된 메타데이터 사용
                page_number = chunk_data["page_number"]
                line_start = chunk_data["line_start"]
                line_end = chunk_data["line_end"]

                chunk_metadata = {
                    "document_id": document_id,
                    "chunk_index": i,
                    "chunk_text": chunk,
                    "file_name": file_name,
                    "file_path": stored_file_path,
                    "file_type": file_extension,
                    "processed_at": datetime.now(TIMEZONE).isoformat(),
                    "chunk_size": len(chunk),
                    "total_chunks": len(chunks),
                    "line_start": line_start,
                    "line_end": line_end,
                    "page_number": page_number,
                    "global_start": chunk_data.get("global_start", -1),
                    "global_end": chunk_data.get("global_end", -1)
                }
                # 각 청크별로 final_metadata를 복사하여 개별 payload 생성 (mutable 공유 방지)
                payload = dict(final_metadata) if isinstance(final_metadata, dict) else {}
                payload.update(chunk_metadata)

                # LLM 메타데이터가 있는 경우에만 업데이트
                if llm_gen_chunk_metadata and isinstance(llm_gen_chunk_metadata, dict):
                    payload.update(llm_gen_chunk_metadata)

                # LLM 메타데이터 생성 여부 표시
                if use_llm_metadata and await self.metadata_generator.is_enabled():
                    payload['llm_metadata_generated'] = True

                # 각 청크마다 새로운 UUID 생성
                chunk_id = str(uuid.uuid4())

                point = {
                    "id": chunk_id,
                    "vector": embedding,
                    "payload": payload
                }
                points.append(point)

            # 벡터 DB에 삽입
            operation_info = self.vector_manager.insert_points(collection_name, points)

            # 컬렉션 메타데이터 업데이트 (문서 수 증가)
            self.vector_manager.update_collection_document_count(collection_name, 1)

            # LLM 메타데이터 생성 여부 로깅
            for point in points:
                payload = point["payload"]

                # 리스트 타입 필드들을 안전하게 처리
                def safe_list_to_string(value):
                    """리스트를 PostgreSQL이 인식할 수 있는 문자열로 변환"""
                    if isinstance(value, list):
                        if not value:  # 빈 리스트인 경우
                            return None
                        # 각 요소를 문자열로 변환하고 PostgreSQL 배열 형식으로 포맷
                        escaped_items = [str(item).replace('"', '""') for item in value]
                        return "{" + ",".join(f'"{item}"' for item in escaped_items) + "}"
                    elif isinstance(value, str) and value.startswith('[') and value.endswith(']'):
                        try:
                            # JSON 문자열을 파싱해서 리스트로 변환 후 다시 PostgreSQL 형식으로
                            import json
                            parsed_list = json.loads(value)
                            if isinstance(parsed_list, list):
                                if not parsed_list:  # 빈 리스트인 경우
                                    return None
                                escaped_items = [str(item).replace('"', '""') for item in parsed_list]
                                return "{" + ",".join(f'"{item}"' for item in escaped_items) + "}"
                        except (json.JSONDecodeError, Exception):
                            pass
                    return value

                def safe_parse_to_list(value):
                    """값을 안전하게 리스트로 파싱"""
                    if isinstance(value, list):
                        return value
                    elif isinstance(value, str) and value.startswith('[') and value.endswith(']'):
                        try:
                            import json
                            parsed_list = json.loads(value)
                            if isinstance(parsed_list, list):
                                return parsed_list
                        except (json.JSONDecodeError, Exception):
                            pass
                    return []

                provider_info = self.embeddings_client.get_provider_info()
                embedding_provider=provider_info.get("provider", "unknown")
                if embedding_provider == 'openai':
                    embedding_model_name = self.config_composer.get_config_by_name('OPENAI_EMBEDDING_MODEL_NAME').value
                elif embedding_provider == 'huggingface':
                    embedding_model_name = self.config_composer.get_config_by_name('HUGGINGFACE_EMBEDDING_MODEL_NAME').value
                elif embedding_provider == 'custom_http':
                    embedding_model_name = self.config_composer.get_config_by_name('CUSTOM_EMBEDDING_MODEL_NAME').value
                else:
                    embedding_model_name = provider_info.get("model_name", "unknown")

                vectordb_chunk_meta = VectorDBChunkMeta(
                    user_id=user_id,
                    collection_name=collection_name,
                    file_name=file_name,
                    chunk_id=point['id'],
                    chunk_text=payload.get("chunk_text"),
                    chunk_index=payload.get("chunk_index"),
                    total_chunks=payload.get("total_chunks"),
                    chunk_size=payload.get("chunk_size"),
                    summary=payload.get("summary"),
                    keywords=safe_list_to_string(payload.get("keywords")),
                    topics=safe_list_to_string(payload.get("topics")),
                    entities=safe_list_to_string(payload.get("entities")),
                    sentiment=payload.get("sentiment"),
                    document_id=document_id,
                    document_type=payload.get("document_type"),
                    language=payload.get("language"),
                    complexity_level=payload.get("complexity_level"),
                    main_concepts=safe_list_to_string(payload.get("main_concepts")),
                    embedding_provider=embedding_provider,
                    embedding_model_name=embedding_model_name,
                    embedding_dimension=provider_info.get("dimension", 0)
                )
                app_db.insert(vectordb_chunk_meta)
                app_db.insert(VectorDBChunkEdge(
                    user_id=user_id,
                    collection_name=collection_name,
                    target=point['id'],
                    source=collection_name,
                    relation_type="chunk",
                ))
                # document_type이 존재할 때만 엣지로 저장 (NULL 삽입 방지)
                doc_type_val = payload.get("document_type")
                if doc_type_val and doc_type_val.strip() != "기타":
                    app_db.insert(VectorDBChunkEdge(
                        user_id=user_id,
                        collection_name=collection_name,
                        target=doc_type_val,
                        source=point['id'],
                        relation_type="document_type"
                    ))
                for keyword in safe_parse_to_list(payload.get("keywords")):
                    app_db.insert(VectorDBChunkEdge(
                        user_id=user_id,
                        collection_name=collection_name,
                        target=keyword,
                        source=point['id'],
                        relation_type="keyword"
                    ))
                for topic in safe_parse_to_list(payload.get("topics")):
                    app_db.insert(VectorDBChunkEdge(
                        user_id=user_id,
                        collection_name=collection_name,
                        target=topic,
                        source=point['id'],
                        relation_type="topic"
                    ))
                for entity in safe_parse_to_list(payload.get("entities")):
                    app_db.insert(VectorDBChunkEdge(
                        user_id=user_id,
                        collection_name=collection_name,
                        target=entity,
                        source=point['id'],
                        relation_type="entity"
                    ))
                for main_concept in safe_parse_to_list(payload.get("main_concepts")):
                    app_db.insert(VectorDBChunkEdge(
                        user_id=user_id,
                        collection_name=collection_name,
                        target=main_concept,
                        source=point['id'],
                        relation_type="main_concept"
                    ))

            vector_db_collection_meta = app_db.find_by_condition(VectorDB, {'collection_name': collection_name})[0]
            if vector_db_collection_meta.init_embedding_model == None:
                vector_db_collection_meta.init_embedding_model = embedding_model_name
                app_db.update(vector_db_collection_meta)

            llm_enabled = use_llm_metadata and await self.metadata_generator.is_enabled()
            logger.info(f"use_llm_metadata: {use_llm_metadata}, LLM metadata enabled: {await self.metadata_generator.is_enabled()}")
            logger.info(f"Document processed successfully: {len(chunks)} chunks inserted (LLM metadata: {llm_enabled})")

            return {
                "message": "Document processed successfully",
                "document_id": document_id,
                "file_name": file_name,
                "chunks_created": len(chunks),
                "llm_metadata_generated": llm_enabled,
                "operation_id": operation_info.get("operation_id"),
                "status": operation_info.get("status")
            }

        except Exception as e:
            logger.error(f"Failed to process document {file_path}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to process document: {e}")

    async def search_documents(self, collection_name: str, query_text: str,
                             limit: int = 5, score_threshold: float = 0.7,
                             filter_criteria: Dict[str, Any] = None,
                             rerank: bool = False, rerank_top_k: int = 20) -> Dict[str, Any]:
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
                limit=max(limit, rerank_top_k) if rerank else limit,
                score_threshold=score_threshold,
                query_filter=search_filter
            )

            # 결과 포맷팅
            results = []
            for hit in search_results:
                payload = hit.payload or {}
                result = {
                    "id": hit.id,
                    "score": hit.score,
                    "document_id": payload.get("document_id"),
                    "chunk_index": payload.get("chunk_index"),
                    "chunk_text": payload.get("chunk_text"),
                    "file_name": payload.get("file_name"),
                    "file_path": payload.get("file_path"),
                    "file_type": payload.get("file_type"),
                    "line_start": payload.get("line_start"),
                    "line_end": payload.get("line_end"),
                    "page_number": payload.get("page_number"),
                    "metadata": {k: v for k, v in payload.items()
                               if k not in ["chunk_text", "document_id", "chunk_index", "file_name", "file_type", "file_path", "line_start", "line_end", "page_number"]}
                }
                results.append(result)

            # Optional reranking using embeddings (if requested and embedding client available)
            if rerank and results:
                # normalize rerank flag if passed as string
                try:
                    if isinstance(rerank, str):
                        rerank = rerank.lower() in ('true', '1', 'yes')
                except Exception:
                    pass

                if rerank:
                    try:
                        # Cross-encoder 기반 재순위
                        from service.embedding.cross_encoder_service import get_cross_encoder_reranker

                        # take top-k candidates to rerank
                        top_candidates = results[:rerank_top_k]
                        candidate_texts = [r.get("chunk_text") or "" for r in top_candidates]

                        logger.info(f"Starting cross-encoder rerank for {len(candidate_texts)} candidates (top_k={rerank_top_k})")

                        # Cross-encoder로 재순위 (타임아웃 적용)
                        try:
                            import asyncio as _asyncio

                            def run_cross_encoder():
                                cross_encoder = get_cross_encoder_reranker()
                                return cross_encoder.rerank(query_text, candidate_texts, top_k=len(candidate_texts))

                            timeout_sec = 10

                            # Cross-encoder 실행 (동기 → 비동기 래핑)
                            rerank_results = await _asyncio.wait_for(
                                _asyncio.to_thread(run_cross_encoder),
                                timeout=timeout_sec
                            )

                            # 재순위 결과 적용
                            if rerank_results:
                                # rerank_results는 [(index, score), ...] 형태
                                reranked_candidates = []
                                for idx, rerank_score in rerank_results:
                                    candidate = top_candidates[idx].copy()
                                    candidate["rerank_score"] = rerank_score
                                    reranked_candidates.append(candidate)

                                # 재순위된 결과와 나머지 결합
                                results = reranked_candidates + results[rerank_top_k:]
                                results = results[:limit]

                                logger.info(f"Cross-encoder rerank completed. Top rerank score: {rerank_results[0][1]:.4f}")
                            else:
                                logger.warning("Cross-encoder returned empty results")

                        except Exception as rerank_e:
                            logger.warning(f"Cross-encoder rerank failed or timed out: {rerank_e}")
                            # 실패 시 원본 결과 유지

                    except Exception as e:
                        logger.warning(f"Reranking setup failed: {e}")

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
                        "file_path": payload.get("file_path", payload.get("file_name", "unknown")),
                        "file_type": payload.get("file_type", "unknown"),
                        "processed_at": payload.get("processed_at"),
                        "total_chunks": payload.get("total_chunks", 0),
                        "chunks": [],
                        "metadata": {},
                        "llm_metadata_generated": payload.get("llm_metadata_generated", False)
                    }

                    # 사용자 정의 메타데이터 추출 (시스템 필드 제외)
                    system_fields = {
                        "document_id", "chunk_index", "chunk_text", "file_name",
                        "file_path", "file_type", "processed_at", "chunk_size", "total_chunks",
                        "line_start", "line_end", "page_number"
                    }
                    for key, value in payload.items():
                        if key not in system_fields:
                            documents[document_id]["metadata"][key] = value

                # 청크 정보 추가
                chunk_info = {
                    "chunk_id": str(point.id),
                    "chunk_index": payload.get("chunk_index", 0),
                    "chunk_size": payload.get("chunk_size", 0),
                    "line_start": payload.get("line_start"),
                    "line_end": payload.get("line_end"),
                    "page_number": payload.get("page_number"),
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

            first_point = points[0]
            payload = first_point.payload

            document_info = {
                "document_id": document_id,
                "file_name": payload.get("file_name", "unknown"),
                "file_path": payload.get("file_path", payload.get("file_name", "unknown")),
                "file_type": payload.get("file_type", "unknown"),
                "processed_at": payload.get("processed_at"),
                "total_chunks": len(points),
                "llm_metadata_generated": payload.get("llm_metadata_generated", False),
                "metadata": {}
            }

            # 사용자 정의 메타데이터 추출
            system_fields = {
                "document_id", "chunk_index", "chunk_text", "file_name",
                "file_path", "file_type", "processed_at", "chunk_size", "total_chunks",
                "line_start", "line_end", "page_number"
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
                    "line_start": chunk_payload.get("line_start"),
                    "line_end": chunk_payload.get("line_end"),
                    "page_number": chunk_payload.get("page_number"),
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

            point_ids = [str(point.id) for point in points]
            operation_info = self.vector_manager.delete_points(collection_name, point_ids)
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
                "provider": self.config_composer.get_config_by_name('EMBEDDING_PROVIDER').value,
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
                "provider": self.config_composer.get_config_by_name('EMBEDDING_PROVIDER').value,
                "error": str(e),
                "available": False
            }

    async def remake_collection(self, user_id: int, app_db, collection_name: str, new_embedding_model: str = None) -> Dict[str, Any]:
        """기존 컬렉션을 새로운 임베딩 모델로 재생성

        기존 컬렉션의 모든 문서를 보존하면서 새로운 임베딩 차원으로 컬렉션을 재생성합니다.

        Args:
            user_id: 사용자 ID
            app_db: 데이터베이스 매니저
            collection_name: 재생성할 컬렉션 이름
            new_embedding_model: 새로운 임베딩 모델명 (None이면 현재 설정 사용)

        Returns:
            리메이크 결과 정보

        Raises:
            HTTPException: 리메이크 실패
        """
        if not self.vector_manager.is_connected():
            raise HTTPException(status_code=500, detail="Vector database not connected")

        new_collection_name = None

        try:
            # 1. 기존 컬렉션 정보 조회
            logger.info(f"Starting remake for collection '{collection_name}'")

            # 데이터베이스에서 컬렉션 메타데이터 조회
            vector_db_meta = app_db.find_by_condition(VectorDB, {
                'user_id': user_id,
                'collection_name': collection_name
            })

            if not vector_db_meta:
                raise HTTPException(status_code=404, detail="Collection not found or not owned by user")

            vector_db_meta = vector_db_meta[0]

            # Qdrant에서 컬렉션 정보 확인
            try:
                collection_info = self.vector_manager.get_collection_info(collection_name)
            except Exception as e:
                raise HTTPException(status_code=404, detail=f"Collection not found in vector database: {e}")

            # 2. 기존 컬렉션의 모든 문서 청크 메타데이터 조회
            chunk_metas = app_db.find_by_condition(VectorDBChunkMeta, {
                'user_id': user_id,
                'collection_name': collection_name
            }, limit=10000, return_list=True)

            if not chunk_metas:
                logger.warning(f"No document chunks found in collection '{collection_name}'")

            # 3. 기존 컬렉션의 모든 실제 데이터 포인트 조회 (메타데이터 포인트 제외)
            logger.info(f"Retrieving all document chunks from collection '{collection_name}'")

            # 메타데이터 포인트 제외 필터
            search_filter = Filter(
                must_not=[
                    FieldCondition(
                        key="type",
                        match=MatchValue(value="collection_metadata")
                    )
                ]
            )

            all_points = []
            offset = None

            while True:
                response = self.vector_manager.client.scroll(
                    collection_name=collection_name,
                    scroll_filter=search_filter,
                    limit=1000,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False  # 벡터는 나중에 새로 생성
                )

                if not response[0]:
                    break

                all_points.extend(response[0])
                offset = response[1]

                if offset is None:
                    break

            logger.info(f"Retrieved {len(all_points)} document chunks from collection")

            # 4. 새로운 임베딩 설정 확인
            provider_info = self.embeddings_client.get_provider_info()
            embedding_provider = provider_info.get("provider", "unknown")
            if embedding_provider == 'openai':
                new_embedding_model = self.config_composer.get_config_by_name('OPENAI_EMBEDDING_MODEL_NAME').value
            elif embedding_provider == 'huggingface':
                new_embedding_model = self.config_composer.get_config_by_name('HUGGINGFACE_EMBEDDING_MODEL_NAME').value
            elif embedding_provider == 'custom_http':
                new_embedding_model = self.config_composer.get_config_by_name('CUSTOM_EMBEDDING_MODEL_NAME').value
            else:
                new_embedding_model = provider_info.get("model_name", "unknown")

            logger.info(f"New embedding model: {new_embedding_model}")

            # 5. 새로운 컬렉션 이름 생성 및 차원 감지
            new_collection_name = f"{vector_db_meta.collection_make_name}_{uuid.uuid4().hex}"
            logger.info(f"Creating new collection '{new_collection_name}'")

            # 새로운 임베딩 차원 감지
            try:
                sample_text = "sample text for dimension detection"
                sample_embedding = await self.generate_embeddings([sample_text])
                current_vector_size = len(sample_embedding[0])
                logger.info(f"Detected new vector dimension: {current_vector_size}")
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to detect new embedding dimension: {e}")

            # 기존 distance metric 가져오기
            old_distance = collection_info.get("config", {}).get("distance", "Cosine")
            # Qdrant의 대소문자 형식을 VectorManager 형식으로 변환
            distance_mapping = {
                "COSINE": "Cosine",
                "EUCLID": "Euclid",
                "DOT": "Dot"
            }
            distance_metric = distance_mapping.get(old_distance.upper(), "Cosine")

            # 새로운 컬렉션 생성
            self.vector_manager.create_collection(
                collection_name=new_collection_name,
                vector_size=current_vector_size,
                distance=distance_metric
            )

            # 6. 모든 문서 청크를 새로운 임베딩으로 다시 생성하여 새 컬렉션에 저장
            logger.info("Re-embedding all document chunks with new model")

            from qdrant_client.models import PointStruct

            batch_size = 50
            total_processed = 0

            for i in range(0, len(all_points), batch_size):
                batch_points = all_points[i:i + batch_size]
                batch_texts = [point.payload.get('chunk_text', '') for point in batch_points]

                # 새 임베딩 생성
                try:
                    new_embeddings = await self.generate_embeddings(batch_texts)
                except Exception as e:
                    logger.error(f"Failed to generate embeddings for batch {i//batch_size + 1}: {e}")
                    # 새 컬렉션 정리
                    try:
                        self.vector_manager.delete_collection(new_collection_name)
                    except:
                        pass
                    raise HTTPException(status_code=500, detail=f"Failed to generate new embeddings: {e}")

                # 새로운 포인트 생성 (딕셔너리 형태로)
                new_points = []
                for j, point in enumerate(batch_points):
                    new_point = {
                        "id": point.id,
                        "vector": new_embeddings[j],
                        "payload": point.payload
                    }
                    new_points.append(new_point)

                # 새 컬렉션에 삽입
                try:
                    self.vector_manager.insert_points(new_collection_name, new_points)
                    total_processed += len(new_points)

                    if (i // batch_size + 1) % 10 == 0:
                        logger.info(f"Processed {total_processed}/{len(all_points)} chunks")
                except Exception as e:
                    logger.error(f"Failed to insert batch {i//batch_size + 1} into new collection: {e}")
                    # 새 컬렉션 정리
                    try:
                        self.vector_manager.delete_collection(new_collection_name)
                    except:
                        pass
                    raise HTTPException(status_code=500, detail=f"Failed to insert points into new collection: {e}")

            logger.info(f"Successfully re-embedded all {total_processed} chunks")

            # 7. 원본 컬렉션 삭제 (새 컬렉션 생성이 성공한 후)
            logger.info(f"Deleting original collection '{collection_name}'")
            try:
                self.vector_manager.delete_collection(collection_name)
            except Exception as e:
                logger.error(f"Failed to delete original collection: {e}")
                # 원본 삭제 실패 시 새 컬렉션 유지하고 경고만 출력
                logger.warning(f"Original collection '{collection_name}' could not be deleted, but new collection '{new_collection_name}' is ready")

            final_collection_name = new_collection_name

            # 8. 데이터베이스 메타데이터 업데이트
            logger.info("Updating database metadata")

            # VectorDB 메타데이터 업데이트
            vector_db_meta.collection_name = final_collection_name
            vector_db_meta.vector_size = current_vector_size
            vector_db_meta.init_embedding_model = new_embedding_model
            vector_db_meta.updated_at = datetime.now(TIMEZONE)
            app_db.update(vector_db_meta)

            # VectorDBChunkMeta 업데이트 (update_list_columns 사용)
            embedding_provider = provider_info.get("provider", "unknown")
            embedding_dimension = provider_info.get("dimension", current_vector_size)

            # 청크 메타데이터 업데이트
            updates = {
                'collection_name': final_collection_name,
                'embedding_provider': embedding_provider,
                'embedding_model_name': new_embedding_model,
                'embedding_dimension': embedding_dimension
            }
            conditions = {
                'user_id': user_id,
                'collection_name': collection_name  # 원본 컬렉션 이름으로 조건 설정
            }

            app_db.update_list_columns(VectorDBChunkMeta, updates, conditions)

            # VectorDBChunkEdge도 업데이트 (존재하는 경우)
            edge_updates = {
                'collection_name': final_collection_name
            }
            app_db.update_list_columns(VectorDBChunkEdge, edge_updates, conditions)

            # 9. 처리된 고유 문서 수 계산
            unique_document_ids = set()
            for point in all_points:
                if 'document_id' in point.payload:
                    unique_document_ids.add(point.payload['document_id'])

            logger.info(f"Collection '{collection_name}' remade successfully as '{final_collection_name}'")

            return {
                "message": f"Collection remade successfully",
                "old_collection_name": collection_name,
                "new_collection_name": final_collection_name,
                "chunks_processed": total_processed,
                "documents_count": len(unique_document_ids),
                "old_vector_size": collection_info.get("config", {}).get("vector_size", "unknown"),
                "new_vector_size": current_vector_size,
                "old_embedding_model": vector_db_meta.init_embedding_model or "unknown",
                "new_embedding_model": new_embedding_model,
                "remake_timestamp": datetime.now(TIMEZONE).isoformat()
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to remake collection '{collection_name}': {e}")
            # 에러 발생 시 새 컬렉션 정리
            if new_collection_name:
                try:
                    self.vector_manager.delete_collection(new_collection_name)
                except:
                    pass
            raise HTTPException(status_code=500, detail=f"Failed to remake collection: {str(e)}")

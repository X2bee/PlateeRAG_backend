"""
벡터 DB 관리 모듈

이 모듈은 Qdrant 벡터 데이터베이스의 컬렉션과 포인트를 관리하는 기능을 제공합니다.
컬렉션 생성/삭제, 포인트 삽입/검색/삭제, 메타데이터 관리 등을 담당합니다.
"""

import logging
import uuid
from datetime import datetime
from typing import List, Dict, Any, Union, Optional, Tuple
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct, Filter, FieldCondition,
    Range, MatchValue, CollectionInfo, UpdateResult, ScoredPoint
)

logger = logging.getLogger("vector-manager")
class VectorManager:
    """벡터 DB 관리를 담당하는 클래스"""
    def __init__(self, vectordb_config):
        """VectorManager 초기화

        Args:
            vectordb_config: 벡터 DB 설정 객체
        """
        self.config = vectordb_config
        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        """Qdrant 클라이언트 초기화"""
        try:
            host = self.config.QDRANT_HOST.value
            port = self.config.QDRANT_PORT.value
            api_key = self.config.QDRANT_API_KEY.value
            use_grpc = self.config.QDRANT_USE_GRPC.value
            grpc_port = self.config.QDRANT_GRPC_PORT.value

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
            raise

    def is_connected(self) -> bool:
        """Qdrant 클라이언트 연결 상태 확인"""
        return self.client is not None

    def cleanup(self):
        """VectorManager 리소스 정리"""
        try:
            if self.client:
                if hasattr(self.client, 'close'):
                    self.client.close()
                    logger.info("Qdrant client closed")
                elif hasattr(self.client, '_client') and hasattr(self.client._client, 'close'):
                    self.client._client.close()
                    logger.info("Qdrant underlying client closed")

                self.client = None
                logger.info("VectorManager cleanup completed")
        except Exception as e:
            logger.warning(f"Error during VectorManager cleanup: {e}")
        finally:
            self.client = None

    def ensure_connected(self):
        """연결 상태 확인 및 필요시 재연결"""
        if not self.is_connected():
            self._initialize_client()
        if not self.is_connected():
            raise Exception("Qdrant client not initialized")

    def create_collection(self, collection_name: str, vector_size: int,
                         distance: str = "Cosine", description: str = None,
                         metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """컬렉션 생성 (메타데이터 지원)

        Args:
            collection_name: 컬렉션 이름
            vector_size: 벡터 차원 수
            distance: 거리 메트릭 ("Cosine", "Euclidean", "Dot")
            description: 컬렉션 설명
            metadata: 사용자 정의 메타데이터

        Returns:
            생성 결과 정보

        Raises:
            Exception: 컬렉션 생성 실패
        """
        self.ensure_connected()

        try:
            # 컬렉션 존재 여부 확인
            try:
                existing_collection = self.client.get_collection(collection_name)
                if existing_collection:
                    # 이미 존재하는 컬렉션 정보 반환
                    logger.info(f"Collection '{collection_name}' already exists")
                    return {
                        "message": f"Collection '{collection_name}' already exists",
                        "collection_id": collection_name,
                        "status": "already_exists",
                        "existing_info": {
                            "vector_size": existing_collection.config.params.vectors.size,
                            "distance_metric": existing_collection.config.params.vectors.distance.name,
                            "vectors_count": getattr(existing_collection, 'vectors_count', 0)
                        }
                    }
            except Exception:
                # 컬렉션이 존재하지 않으면 계속 진행
                pass

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

                # 메타데이터 포인트용 UUID 생성
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
                "status": "created",
                "metadata": {
                    "description": description,
                    "custom_metadata": metadata,
                    "vector_size": vector_size,
                    "distance_metric": distance
                }
            }

        except Exception as e:
            logger.error(f"Failed to create collection '{collection_name}': {e}")
            raise

    def delete_collection(self, collection_name: str) -> Dict[str, Any]:
        """컬렉션 삭제

        Args:
            collection_name: 삭제할 컬렉션 이름

        Returns:
            삭제 결과 정보

        Raises:
            Exception: 컬렉션 삭제 실패
        """
        self.ensure_connected()

        try:
            self.client.delete_collection(collection_name)
            logger.info(f"Collection '{collection_name}' deleted successfully")
            return {"message": f"Collection '{collection_name}' deleted successfully", "status": "success"}

        except Exception as e:
            logger.error(f"Failed to delete collection '{collection_name}': {e}")
            raise

    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """컬렉션 정보 조회

        Args:
            collection_name: 조회할 컬렉션 이름

        Returns:
            컬렉션 정보

        Raises:
            Exception: 컬렉션 조회 실패
        """
        self.ensure_connected()

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
            raise

    def list_collections(self) -> Dict[str, Any]:
        """모든 컬렉션 목록 조회

        Returns:
            컬렉션 목록

        Raises:
            Exception: 컬렉션 목록 조회 실패
        """
        self.ensure_connected()

        try:
            collections = self.client.get_collections()
            return {
                "collections": [collection.name for collection in collections.collections]
            }

        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            raise

    def insert_points(self, collection_name: str, points: List[Dict[str, Any]]) -> Dict[str, Any]:
        """벡터 포인트 삽입

        Args:
            collection_name: 대상 컬렉션 이름
            points: 삽입할 포인트 리스트. 각 포인트는 다음 키를 가져야 함:
                   - vector: List[float] (필수)
                   - payload: Dict[str, Any] (선택)
                   - id: Union[str, int] (선택, 없으면 자동 생성)

        Returns:
            삽입 결과 정보

        Raises:
            Exception: 포인트 삽입 실패
        """
        self.ensure_connected()

        try:
            qdrant_points = []
            for point in points:
                # point.id가 없으면 새 UUID 생성, 있으면 유효한 형식인지 확인
                if "id" not in point or point["id"] is None:
                    point_id = str(uuid.uuid4())
                else:
                    # 기존 ID가 정수이거나 유효한 UUID인지 확인
                    try:
                        # 정수로 변환 시도
                        point_id = int(point["id"])
                    except (ValueError, TypeError):
                        try:
                            # UUID 형식 검증 시도
                            uuid.UUID(str(point["id"]))
                            point_id = str(point["id"])
                        except ValueError:
                            # 유효하지 않은 형식이면 새 UUID 생성
                            logger.warning(f"Invalid point ID format: {point['id']}, generating new UUID")
                            point_id = str(uuid.uuid4())

                qdrant_points.append(
                    PointStruct(
                        id=point_id,
                        vector=point["vector"],
                        payload=point.get("payload", {})
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
            raise

    def search_points(self, collection_name: str, query_vector: List[float],
                     limit: int = 10, score_threshold: float = None,
                     filter_criteria: Dict[str, Any] = None) -> Dict[str, Any]:
        """벡터 유사도 검색

        Args:
            collection_name: 검색할 컬렉션 이름
            query_vector: 쿼리 벡터
            limit: 반환할 최대 결과 수
            score_threshold: 최소 유사도 임계값
            filter_criteria: 필터 조건

        Returns:
            검색 결과

        Raises:
            Exception: 검색 실패
        """
        self.ensure_connected()

        try:
            search_filter = None
            if filter_criteria:
                search_filter = self._build_filter(filter_criteria)

            search_results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold,
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
            raise

    def delete_points(self, collection_name: str, point_ids: List[Union[str, int]]) -> Dict[str, Any]:
        """포인트 삭제

        Args:
            collection_name: 대상 컬렉션 이름
            point_ids: 삭제할 포인트 ID 리스트

        Returns:
            삭제 결과 정보

        Raises:
            Exception: 포인트 삭제 실패
        """
        self.ensure_connected()

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
            raise

    def scroll_points(self, collection_name: str, filter_criteria: Dict[str, Any] = None,
                     limit: int = 100, offset: Optional[str] = None,
                     with_payload: bool = True, with_vectors: bool = False) -> Tuple[List, Optional[str]]:
        """포인트 스크롤 조회

        Args:
            collection_name: 조회할 컬렉션 이름
            filter_criteria: 필터 조건
            limit: 한 번에 조회할 포인트 수
            offset: 시작 오프셋
            with_payload: 페이로드 포함 여부
            with_vectors: 벡터 포함 여부

        Returns:
            (포인트 리스트, 다음 오프셋) 튜플

        Raises:
            Exception: 스크롤 조회 실패
        """
        self.ensure_connected()

        try:
            search_filter = None
            if filter_criteria:
                search_filter = self._build_filter(filter_criteria)

            scroll_result = self.client.scroll(
                collection_name=collection_name,
                scroll_filter=search_filter,
                limit=limit,
                offset=offset,
                with_payload=with_payload,
                with_vectors=with_vectors
            )

            points, next_offset = scroll_result
            return points, next_offset

        except Exception as e:
            logger.error(f"Failed to scroll points in '{collection_name}': {e}")
            raise

    def update_collection_document_count(self, collection_name: str, increment: int):
        """컬렉션의 문서 수 업데이트

        Args:
            collection_name: 대상 컬렉션 이름
            increment: 증감량 (양수: 증가, 음수: 감소)
        """
        try:
            # 기존 메타데이터 조회
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
                new_count = max(0, current_count + increment)  # 음수 방지

                # 메타데이터 업데이트를 위한 새로운 포인트 생성
                updated_payload = metadata_point.payload.copy()
                updated_payload["document_count"] = new_count
                updated_payload["last_updated"] = datetime.now().isoformat()

                # 메타데이터 포인트의 벡터 확인 및 수정
                if metadata_point.vector is None or len(metadata_point.vector) == 0:
                    # 메타데이터 포인트의 벡터가 없으면 현재 설정된 차원으로 더미 벡터 생성
                    current_dimension = self.config.VECTOR_DIMENSION.value
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

                logger.info(f"Updated document count for collection '{collection_name}': {current_count} -> {new_count}")

        except Exception as e:
            logger.warning(f"Failed to update collection document count: {e}")
            # 실패해도 전체 프로세스는 계속 진행

    def _build_filter(self, filter_criteria: Dict[str, Any]) -> Filter:
        """필터 조건 빌드

        Args:
            filter_criteria: 필터 조건 딕셔너리

        Returns:
            Qdrant Filter 객체
        """
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

        return Filter(must=conditions) if conditions else None

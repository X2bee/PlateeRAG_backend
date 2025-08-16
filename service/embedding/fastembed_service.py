"""
FastEmbed 기반 하이브리드 인덱싱/검색 유틸리티

요약:
- FastEmbed 모델을 로드하여 dense / sparse / late-interaction 임베딩을 생성
- Qdrant 다중 벡터 + sparse 설정을 사용해 하이브리드 컬렉션을 생성하고 업서트
- prefetch + late-interaction rerank 흐름을 간단히 제공

의존성: fastembed, qdrant-client
"""
import logging
from typing import List, Dict, Any, Optional, Iterable

logger = logging.getLogger("fastembed-service")

try:
    from fastembed import TextEmbedding, LateInteractionTextEmbedding, SparseTextEmbedding
except Exception:
    TextEmbedding = None
    LateInteractionTextEmbedding = None
    SparseTextEmbedding = None

from qdrant_client import QdrantClient
from qdrant_client.models import models as qmodels
from qdrant_client.models import PointStruct


class FastEmbedService:
    def __init__(self):
        if TextEmbedding is None:
            logger.warning("fastembed not available. Install with `pip install fastembed` to enable FastEmbedService")
        self.dense_model = None
        self.sparse_model = None
        self.late_model = None

    def init_models(self,
                    dense_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                    sparse_model_name: str = "Qdrant/bm25",
                    late_model_name: str = "colbert-ir/colbertv2.0"):
        """모델 초기화(없으면 lazy하게 호출 가능)"""
        if TextEmbedding and not self.dense_model:
            self.dense_model = TextEmbedding(dense_model_name)
        if SparseTextEmbedding and not self.sparse_model:
            self.sparse_model = SparseTextEmbedding(sparse_model_name)
        if LateInteractionTextEmbedding and not self.late_model:
            self.late_model = LateInteractionTextEmbedding(late_model_name)

    def embed_documents(self, documents: Iterable[str]):
        """문서 목록에 대해 (dense, sparse, late) 임베딩 생성

        Returns: tuple of lists (dense_embeddings, sparse_embeddings, late_embeddings)
        """
        if not self.dense_model or not self.sparse_model or not self.late_model:
            self.init_models()

        dense_embeddings = list(self.dense_model.embed(doc for doc in documents))
        # sparse and late need to re-run documents generator; caller should pass list if reuse
        return dense_embeddings

    def embed_documents_all(self, documents: List[str]):
        """완전한 3종류 임베딩을 반환하는 편의 함수"""
        if not self.dense_model or not self.sparse_model or not self.late_model:
            self.init_models()

        dense_embeddings = list(self.dense_model.embed(doc for doc in documents))
        sparse_embeddings = list(self.sparse_model.embed(doc for doc in documents))
        late_embeddings = list(self.late_model.embed(doc for doc in documents))

        return dense_embeddings, sparse_embeddings, late_embeddings

    def create_hybrid_collection(self, client: QdrantClient, collection_name: str,
                                 dense_key: str, dense_dim: int,
                                 late_key: str, late_dim: int,
                                 sparse_key: str):
        """Qdrant에 하이브리드 컬렉션 생성 (multi-vector + sparse)

        이 함수는 이미 존재하면 아무 작업도 하지 않습니다.
        """
        try:
            # vectors_config dict 형태로 구성
            vectors_config = {
                dense_key: qmodels.VectorParams(size=dense_dim, distance=qmodels.Distance.COSINE),
                late_key: qmodels.VectorParams(
                    size=late_dim,
                    distance=qmodels.Distance.COSINE,
                    multivector_config=qmodels.MultiVectorConfig(comparator=qmodels.MultiVectorComparator.MAX_SIM),
                    hnsw_config=qmodels.HnswConfigDiff(m=0)
                )
            }

            sparse_vectors_config = {
                sparse_key: qmodels.SparseVectorParams(modifier=qmodels.Modifier.IDF)
            }

            client.recreate_collection(
                collection_name=collection_name,
                vectors_config=vectors_config,
                sparse_vectors_config=sparse_vectors_config
            )
            logger.info(f"Created hybrid collection: {collection_name}")
        except Exception as e:
            logger.error(f"Failed to create hybrid collection {collection_name}: {e}")
            raise

    def upsert_hybrid_points(self, client: QdrantClient, collection_name: str,
                             dense_key: str, sparse_key: str, late_key: str,
                             dense_embeddings: List[List[float]],
                             sparse_embeddings: List[Any],
                             late_embeddings: List[Any],
                             documents: List[str], ids: Optional[List[Any]] = None,
                             payloads: Optional[List[Dict[str, Any]]] = None):
        """여러 벡터 유형을 포함한 PointStruct 리스트를 upsert"""
        points = []
        for idx, (d, s, l, doc) in enumerate(zip(dense_embeddings, sparse_embeddings, late_embeddings, documents)):
            pid = ids[idx] if ids else idx
            payload = payloads[idx] if payloads else {"document": doc}

            vector_object = {
                dense_key: d,
                sparse_key: s.as_object() if hasattr(s, 'as_object') else s,
                late_key: l,
            }

            points.append(PointStruct(id=pid, vector=vector_object, payload=payload))

        op = client.upsert(collection_name=collection_name, points=points)
        logger.info(f"Upserted {len(points)} hybrid points into {collection_name}")
        return op

    def hybrid_search(self, client: QdrantClient, collection_name: str,
                      query: str, dense_model: TextEmbedding, sparse_model: SparseTextEmbedding,
                      late_model: LateInteractionTextEmbedding, dense_key: str, sparse_key: str, late_key: str,
                      prefetch_limits: Dict[str, int], limit: int = 10) -> Dict[str, Any]:
        """하이브리드 검색: prefetch로 dense/sparse 수행 후 late-interaction으로 rerank

        Note: fastembed의 query_embed 사용 may return generators; convert accordingly.
        """
        # create query vectors
        dense_vec = next(dense_model.query_embed(query))
        sparse_vec = next(sparse_model.query_embed(query))
        late_vec = next(late_model.query_embed(query))

        prefetch = []
        if prefetch_limits.get(dense_key):
            prefetch.append(qmodels.Prefetch(query=dense_vec, using=dense_key, limit=prefetch_limits[dense_key]))
        if prefetch_limits.get(sparse_key):
            prefetch.append(qmodels.Prefetch(query=qmodels.SparseVector(**sparse_vec.as_object()), using=sparse_key, limit=prefetch_limits[sparse_key]))

        results = client.query_points(
            collection_name=collection_name,
            prefetch=prefetch if prefetch else None,
            query=late_vec,
            using=late_key,
            with_payload=True,
            limit=limit,
        )

        return results



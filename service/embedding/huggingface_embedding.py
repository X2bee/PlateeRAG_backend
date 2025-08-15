"""
HuggingFace 임베딩 클라이언트 (sentence-transformers 사용)
"""

import asyncio
from typing import List, Dict, Any
import logging
import numpy as np
from service.embedding.base_embedding import BaseEmbedding

logger = logging.getLogger("embeddings.huggingface")

class HuggingFaceEmbedding(BaseEmbedding):
    """HuggingFace sentence-transformers 임베딩 클라이언트"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_name = config.get("model_name", "Qwen/Qwen3-Embedding-0.6B")
        self.api_key = config.get("api_key", "")
        self.model_device = config.get("model_device", "cpu")
        self.model = None
        self._dimension = None

        self._initialize_model()

    def _initialize_model(self):
        """모델 초기화"""
        try:
            from sentence_transformers import SentenceTransformer

            # API key가 있으면 설정
            if self.api_key:
                import os
                os.environ["HUGGINGFACE_HUB_TOKEN"] = self.api_key

            logger.info(f"Loading HuggingFace model: {self.model_name}")

            # 모델 로드 (타임아웃 및 재시도 포함)
            try:
                is_gpu_device = (
                    self.model_device == 'gpu' or
                    self.model_device == 'cuda'
                )
                logger.info(f"Using device: {self.model_device} (GPU: {is_gpu_device})")
                device = 'cuda' if is_gpu_device else 'cpu'
                self.model = SentenceTransformer(
                    self.model_name,
                    device=device
                )
                logger.info(f"Model loaded on ==={device}=== device")
                logger.info(
                    f"HuggingFace model loaded successfully: {self.model_name}"
                )

                # 차원 수 확인
                test_embedding = self.model.encode(["test"], convert_to_numpy=True, show_progress_bar=False)
                self._dimension = test_embedding.shape[1]
                logger.info(f"Model dimension: {self._dimension}")

            except Exception as model_error:
                logger.warning(f"Failed to load model {self.model_name}: {model_error}")

                # 대체 모델 시도
                fallback_models = [
                    "sentence-transformers/all-MiniLM-L6-v2",
                    "sentence-transformers/paraphrase-MiniLM-L6-v2",
                    "sentence-transformers/all-mpnet-base-v2",
                    "BAAI/bge-small-en-v1.5",
                    "BAAI/bge-base-en-v1.5"
                ]

                for fallback_model in fallback_models:
                    if fallback_model != self.model_name:
                        try:
                            logger.info(f"Trying fallback model: {fallback_model}")
                            self.model = SentenceTransformer(fallback_model, device='cpu')
                            self.model_name = fallback_model  # 성공한 모델명으로 업데이트

                            # 차원 수 확인
                            test_embedding = self.model.encode(["test"], convert_to_numpy=True, show_progress_bar=False)
                            self._dimension = test_embedding.shape[1]

                            logger.info(f"Fallback model loaded successfully: {fallback_model}, dimension: {self._dimension}")
                            return

                        except Exception as fallback_error:
                            logger.warning(f"Fallback model {fallback_model} also failed: {fallback_error}")
                            continue

                # 모든 모델이 실패한 경우
                raise Exception(f"All HuggingFace models failed to load")

        except ImportError as import_error:
            logger.error("sentence_transformers package not installed. Run: pip install sentence-transformers")
            logger.error(f"Import error details: {import_error}")
            self.model = None
        except Exception as e:
            logger.error(f"Failed to initialize any HuggingFace model: {e}")
            self.model = None

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """여러 문서를 임베딩으로 변환"""
        if not self.model:
            raise ValueError("HuggingFace model not initialized")

        try:
            # CPU 집약적인 작업을 별도 스레드에서 실행
            embeddings = await asyncio.to_thread(
                self._encode_texts, texts
            )

            logger.info(f"Generated {len(embeddings)} document embeddings")
            return embeddings

        except Exception as e:
            logger.error(f"Failed to generate document embeddings: {e}")
            raise

    async def embed_query(self, text: str) -> List[float]:
        """단일 쿼리를 임베딩으로 변환"""
        if not self.model:
            raise ValueError("HuggingFace model not initialized")

        try:
            # CPU 집약적인 작업을 별도 스레드에서 실행
            embedding = await asyncio.to_thread(
                self._encode_texts, [text]
            )

            logger.info(f"Generated query embedding for text: {text[:50]}...")
            return embedding[0]

        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            raise

    def _encode_texts(self, texts: List[str]) -> List[List[float]]:
        """텍스트 리스트를 임베딩으로 변환 (동기 함수)"""
        # 배치 크기 제한
        batch_size = 32
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # sentence-transformers로 임베딩 생성
            batch_embeddings = self.model.encode(
                batch_texts,
                convert_to_numpy=True,
                normalize_embeddings=True,  # 정규화
                show_progress_bar=False
            )

            # numpy array를 리스트로 변환
            batch_embeddings_list = batch_embeddings.tolist()
            all_embeddings.extend(batch_embeddings_list)

        return all_embeddings

    def get_embedding_dimension(self) -> int:
        """임베딩 차원 수 반환"""
        if self._dimension is not None:
            return self._dimension

        # 모델이 초기화되지 않은 경우 기본값
        return 384  # all-MiniLM-L6-v2의 기본 차원

    async def is_available(self) -> bool:
        """HuggingFace 서비스 사용 가능성 확인"""
        if not self.model:
            return False

        try:
            # 간단한 테스트 임베딩 생성
            test_embedding = await asyncio.to_thread(
                self._encode_texts, ["test"]
            )
            return len(test_embedding) > 0 and len(test_embedding[0]) > 0

        except Exception as e:
            logger.warning(f"HuggingFace embedding service not available: {e}")
            return False

    def get_provider_info(self) -> Dict[str, Any]:
        """HuggingFace 제공자 정보 반환"""
        return {
            "provider": "huggingface",
            "model": self.model_name,
            "dimension": self.get_embedding_dimension(),
            "api_key_configured": bool(self.api_key),
            "available": bool(self.model)
        }

    async def cleanup(self):
        """HuggingFace 모델 리소스 정리"""
        logger.info("Cleaning up HuggingFace embedding client: %s", self.model_name)

        if self.model:
            try:
                # 모델을 CPU로 이동 (GPU 메모리 해제)
                if hasattr(self.model, 'to'):
                    self.model.to('cpu')

                # PyTorch 캐시 정리
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    logger.info("PyTorch GPU cache cleared")
                except ImportError:
                    logger.info("PyTorch not available, skipping GPU cleanup")

                # 모델 객체 정리
                del self.model
                self.model = None

                # 차원 정보 리셋
                self._dimension = None

                logger.info("HuggingFace model cleanup completed")

            except Exception as e:
                logger.warning("Error during HuggingFace model cleanup: %s", e)

        # 부모 클래스 cleanup 호출
        await super().cleanup()

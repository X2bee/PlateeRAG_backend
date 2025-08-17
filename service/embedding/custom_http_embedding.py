"""
Custom HTTP API 임베딩 클라이언트 (vLLM 등)
"""

import asyncio
from typing import List, Dict, Any
import logging
import aiohttp
import requests
import json
from service.embedding.base_embedding import BaseEmbedding

logger = logging.getLogger("embeddings.custom_http")

class CustomHTTPEmbedding(BaseEmbedding):
    """Custom HTTP API 임베딩 클라이언트 (vLLM, FastAPI 등)"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = config.get("url", "http://localhost:8000/v1").rstrip("/")
        self.api_key = config.get("api_key", "")
        self.model = config.get("model", "text-embedding-ada-002")
        self._dimension = None
        if "/embeddings" in self.base_url:
            self.embeddings_endpoint = self.base_url
            # Derive a clean base_root by removing the first occurrence of /embeddings
            parts = self.base_url.split("/embeddings", 1)
            self.base_root = parts[0] if parts and parts[0] else self.base_url
        else:
            self.embeddings_endpoint = f"{self.base_url}/embeddings"
            self.base_root = self.base_url
        
        # HTTP 설정
        self.timeout = aiohttp.ClientTimeout(total=300)  # 5분
        self.headers = {
            "Content-Type": "application/json"
        }
        
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"
        
        logger.info(f"Custom HTTP embedding client initialized: base_url={self.base_root}, embeddings_endpoint={self.embeddings_endpoint}")
    
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """여러 문서를 임베딩으로 변환"""
        try:
            # 배치 크기 제한
            batch_size = 50
            all_embeddings = []
            
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i + batch_size]
                    
                    embeddings = await self._create_embeddings(session, batch_texts)
                    all_embeddings.extend(embeddings)
            
            logger.info(f"Generated {len(all_embeddings)} document embeddings")
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate document embeddings: {e}")
            raise
    
    async def embed_query(self, text: str) -> List[float]:
        """단일 쿼리를 임베딩으로 변환"""
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                embeddings = await self._create_embeddings(session, [text])
                
                if not embeddings:
                    raise ValueError("No embedding returned from server")
                
                logger.info(f"Generated query embedding for text: {text[:50]}...")
                return embeddings[0]
                
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            raise
    
    async def _create_embeddings(self, session: aiohttp.ClientSession, texts: List[str]) -> List[List[float]]:
        """HTTP API를 통해 임베딩 생성"""
        url = self.embeddings_endpoint
        
        payload = {
            "model": self.model,
            "input": texts,
            "encoding_format": "float"
        }
        
        try:
            async with session.post(url, headers=self.headers, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise ValueError(f"HTTP {response.status}: {error_text}")
                
                result = await response.json()
                
                # OpenAI 호환 응답 형식 처리
                if "data" in result:
                    embeddings = [item["embedding"] for item in result["data"]]
                elif "embeddings" in result:
                    embeddings = result["embeddings"]
                else:
                    raise ValueError(f"Unexpected response format: {result}")
                
                # 차원 수 업데이트
                if embeddings and not self._dimension:
                    self._dimension = len(embeddings[0])
                
                return embeddings
                
        except aiohttp.ClientError as e:
            raise ValueError(f"HTTP client error: {e}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON response: {e}")
    
    def get_embedding_dimension(self) -> int:
        """임베딩 차원 수 반환"""
        if self._dimension is not None:
            return self._dimension

        # _dimension이 아직 설정되지 않은 경우, 동기적으로 간단한 프로브를 시도하여 차원 확인
        try:
            url = self.embeddings_endpoint
            payload = {
                "model": self.model,
                "input": ["test"],
                "encoding_format": "float"
            }
            headers = dict(self.headers) if isinstance(self.headers, dict) else {}
            # 짧은 타임아웃으로 동기 요청 수행
            resp = requests.post(url, json=payload, headers=headers, timeout=5)
            if resp.status_code == 200:
                try:
                    result = resp.json()
                except Exception:
                    result = None

                embedding = None
                if isinstance(result, dict):
                    if "data" in result and isinstance(result["data"], list) and result["data"]:
                        embedding = result["data"][0].get("embedding")
                    elif "embeddings" in result and isinstance(result["embeddings"], list) and result["embeddings"]:
                        embedding = result["embeddings"][0]

                if embedding and isinstance(embedding, (list, tuple)):
                    self._dimension = len(embedding)
                    logger.info(f"Detected embedding dimension via sync probe: {self._dimension}")
                    return self._dimension

        except Exception as e:
            logger.debug(f"Sync probe for embedding dimension failed: {e}")

        # 차원 정보를 아직 알 수 없음
        return None
    
    async def is_available(self) -> bool:
        """Custom HTTP API 서비스 사용 가능성 확인"""
        try:
            timeout = aiohttp.ClientTimeout(total=5)  # 짧은 타임아웃
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                # 헬스 체크 먼저 시도 (우선 embeddings endpoint 기반의 health 경로를 확인)
                health_endpoints = [
                    f"{self.embeddings_endpoint}/health",
                    f"{self.base_root}/health",
                    f"{self.base_root}/v1/health",
                    f"{self.base_root}/"
                ]
                
                health_ok = False
                for health_url in health_endpoints:
                    try:
                        async with session.get(health_url) as response:
                            if response.status in [200, 404]:  # 404도 서버가 응답하는 것으로 간주
                                health_ok = True
                                logger.info(f"Server responding at {health_url}")
                                break
                    except Exception:
                        continue
                
                if not health_ok:
                    logger.warning("Server not responding to health checks")
                    return False
                
                # 실제 임베딩 테스트 (간단하게)
                try:
                    test_embedding = await self._create_embeddings(session, ["test"])
                    is_valid = len(test_embedding) > 0 and len(test_embedding[0]) > 0
                    if is_valid:
                        logger.info("Custom HTTP embedding test successful")
                    return is_valid
                except Exception as e:
                    logger.warning(f"Custom HTTP embedding test failed: {e}")
                    return False
                
        except Exception as e:
            logger.warning(f"Custom HTTP embedding service not available: {e}")
            return False
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Custom HTTP 제공자 정보 반환"""
        return {
            "provider": "custom_http",
            "base_url": self.base_root,
            "embeddings_endpoint": self.embeddings_endpoint,
            "model": self.model,
            "dimension": self.get_embedding_dimension(),
            "api_key_configured": bool(self.api_key),
            "available": True  # is_available()로 실제 확인 필요
        }
    
    async def cleanup(self):
        """Custom HTTP 클라이언트 리소스 정리"""
        logger.info("Cleaning up Custom HTTP embedding client: %s", self.embeddings_endpoint)
        
        try:
            # 설정 초기화
            self._dimension = None
            
            # 헤더 정리 (민감한 정보 제거)
            if "Authorization" in self.headers:
                del self.headers["Authorization"]
                
            logger.info("Custom HTTP client cleanup completed")
            
        except Exception as e:
            logger.warning("Error during Custom HTTP client cleanup: %s", e)
        
        # 부모 클래스 cleanup 호출
        await super().cleanup() 
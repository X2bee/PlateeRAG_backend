"""
RAG 서비스 모듈

이 모듈은 RAG(Retrieval-Augmented Generation) 시스템의 핵심 비즈니스 로직을 제공합니다.
문서 처리, 임베딩 생성, 벡터 검색 등의 기능을 조합하여 완전한 RAG 서비스를 구현합니다.
"""
import re
import logging
import uuid
import json
import asyncio
import os
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import HTTPException
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
import gc
import weakref

from service.embedding import EmbeddingFactory
from service.retrieval.document_processor import DocumentProcessor
from service.vector_db.vector_manager import VectorManager
from service.database.models.vectordb import VectorDBChunkEdge, VectorDBChunkMeta

logger = logging.getLogger("rag-service")

# 환경변수에서 타임존 가져오기 (기본값: 서울 시간)
TIMEZONE = ZoneInfo(os.getenv('TIMEZONE', 'Asia/Seoul'))

class DocumentMetadata(BaseModel):
    """문서 메타데이터를 위한 Pydantic 모델"""
    summary: str = Field(description="주어진 데이터의 핵심 요소를 반드시 추출하여 요약문을 작성하십시오. 이 요약문의 길이 제한은 없으며, 반드시 핵심 정보는 모두 포함되어야 합니다. 특히 연령, 성별, 수치적 요소, 계절, 주기성 등의 메타적 요소가 존재하면 이것은 반드시 포함되어야 합니다.")
    keywords: List[str] = Field(description="핵심 키워드들 (1-3개)", min_items=1, max_items=3)
    topics: List[str] = Field(description="주요 주제들 (1-3개)", min_items=1, max_items=3)
    entities: List[str] = Field(description="개체명들 (인명, 지명, 기관명, 상품명 등)", max_items=5)
    sentiment: str = Field(description="문서의 감정 톤", pattern="^(positive|negative|neutral)$")
    document_type: str = Field(description="문서 유형", pattern="^(상품설명서|기술설명서|보고서|매뉴얼|학술논문|뉴스|사회법률|회사내규|지침서|기타)$")
    language: str = Field(description="언어 코드 (ko, en, ja, zh 등)", max_length=1)
    complexity_level: str = Field(description="복잡도 수준", pattern="^(beginner|intermediate|advanced)$")
    main_concepts: List[str] = Field(description="핵심 개념들 (1-3개)", min_items=1, max_items=3)

class LLMMetadataGenerator:
    """LLM 기반 메타데이터 생성 클래스 (설정 변경 자동 감지/재초기화 버전)"""

    def __init__(self, collection_config: Optional[Any] = None):
        self.collection_config = collection_config
        self.llm_client = None

        # 최근 초기화에 사용된 설정 지문(핑거프린트)
        self._fingerprint: Optional[tuple] = None

        logger.info(f"LLMMetadataGenerator initialized with collection_config: {type(collection_config)}")
        if collection_config:
            logger.info(f"Available attributes: {dir(collection_config)}")

    # -------- 외부에서 collection_config 교체 시 호출 --------
    def set_collection_config(self, collection_config: Any):
        """외부에서 collection_config를 바꿀 때 호출 (다음 ensure에서 재초기화 유도)"""
        self.collection_config = collection_config
        self._fingerprint = None  # invalidate to force reinit

    # -------- 설정 접근 유틸 --------
    def _get_config_value(self, attr_name: str, default_value: Any = None) -> Any:
        """collection_config에서 안전하게 값을 가져오기"""
        try:
            if not self.collection_config:
                return default_value

            if hasattr(self.collection_config, attr_name):
                attr = getattr(self.collection_config, attr_name)
                return getattr(attr, "value", attr)
            else:
                logger.warning(f"Attribute '{attr_name}' not found in collection_config")
                return default_value

        except Exception as e:
            logger.error(f"Error getting config value for '{attr_name}': {e}")
            return default_value

    def _compute_fingerprint(self) -> tuple:
        """현재 설정값으로부터 재초기화 필요 여부 판단용 지문 생성"""
        provider = str(self._get_config_value("IMAGE_TEXT_MODEL_PROVIDER", "no_model")).lower()
        model = self._get_config_value("IMAGE_TEXT_MODEL_NAME", "gpt-4o-mini")
        base_url = self._get_config_value("IMAGE_TEXT_BASE_URL", "https://api.openai.com/v1")
        temperature = float(self._get_config_value("IMAGE_TEXT_TEMPERATURE", 0.1))
        api_key_hint = bool(self._get_config_value("IMAGE_TEXT_API_KEY", ""))  # 키 유무만 반영
        return (provider, model, base_url, temperature, api_key_hint)

    # -------- 클라이언트 초기화/활성화 --------
    def _initialize_llm_client(self):
        """LLM 클라이언트 초기화 (설정 변화가 있으면 항상 새로 초기화)"""
        try:
            provider, model, base_url, temperature, _ = self._compute_fingerprint()

            if provider == "no_model":
                logger.info("No LLM model configured - metadata generation disabled")
                self.llm_client = None
                self._fingerprint = (provider, model, base_url, temperature, _)
                return

            api_key = self._get_config_value("IMAGE_TEXT_API_KEY", "")

            logger.info(f"Initializing LLM client with provider={provider}, model={model}, base_url={base_url}")

            if provider in ("openai", "vllm"):
                # langchain_openai는 OpenAI 호환 Chat Completions 엔드포인트(vLLM 포함)에 대응
                from langchain_openai import ChatOpenAI
                self.llm_client = ChatOpenAI(
                    model=model,
                    openai_api_key=api_key or ("dummy" if provider == "vllm" else ""),
                    base_url=base_url,
                    temperature=temperature,
                )
            else:
                logger.error(f"Unsupported LLM provider: {provider}")
                self.llm_client = None
                self._fingerprint = (provider, model, base_url, temperature, _)
                return

            self._fingerprint = (provider, model, base_url, temperature, _)
            logger.info(f"LLM client initialized successfully: {provider} / {model} @ {base_url}")

        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
            self.llm_client = None
            # _fingerprint를 갱신하지 않음 → 다음 ensure에서 재시도

    async def is_enabled(self) -> bool:
        """메타데이터 생성이 활성화되어 있는지 확인"""
        try:
            provider = str(self._get_config_value("IMAGE_TEXT_MODEL_PROVIDER", "no_model")).lower()
            return provider != "no_model"
        except Exception:
            return False

    async def ensure_llm_client(self):
        """
        - 활성화 여부 체크
        - 설정이 바뀌었으면 재초기화
        - 아직 없으면 초기화
        """
        if not await self.is_enabled():
            return

        current_fp = self._compute_fingerprint()
        if (self.llm_client is None) or (self._fingerprint != current_fp):
            logger.info("LLM client missing or config changed → reinitializing")
            self._initialize_llm_client()

    # -------- Prompt/파서 유틸 --------
    def _create_metadata_prompt(self, text: str, existing_metadata: Optional[Dict[str, Any]] = None) -> str:
        """메타데이터 생성을 위한 프롬프트 생성 (JsonOutputParser 사용)"""
        text_sample = text[:3000] if len(text) > 3000 else text

        parser = JsonOutputParser(pydantic_object=DocumentMetadata)
        format_instructions = parser.get_format_instructions()

        # 중괄호 이스케이프 처리
        escaped_instructions = format_instructions.replace("{", "{{").replace("}", "}}")

        prompt = f"""다음 텍스트를 분석해서 구조화된 메타데이터를 생성해주세요:

=== 텍스트 ===
{text_sample}

=== 출력 형식 ===
{escaped_instructions}

=== 추가 지침 ===
- 반드시 한국어(KOREAN), 영어(ENGLISH) 및 숫자 특수문자만 사용할 것
- 감정 분석에서 중립적인 경우 "neutral" 사용
- 문서 유형이 명확하지 않으면 "기타" 사용
- 키워드와 개념은 중복되지 않도록 주의
- 개체명이 없는 경우 빈 배열 반환
- 명시적인 요청이 없다면 summary는 한글로 반환"""

        if existing_metadata:
            prompt += f"\n\n=== 기존 메타데이터 참고 ===\n{existing_metadata}"

        return prompt

    def _validate_parsed_metadata_language(self, parsed_metadata: Dict[str, Any]) -> bool:
        """파싱된 메타데이터의 각 값이 허용된 언어인지 검증"""
        allowed_pattern = r'^[\u0020-\u007F\u00A0-\u00FF\uAC00-\uD7AF\u1100-\u11FF\u3130-\u318F\s\n\r\t]*$'

        def check_value(value):
            if isinstance(value, str):
                if not re.match(allowed_pattern, value):
                    chinese_chars = re.findall(r'[\u4e00-\u9fff]', value)
                    japanese_chars = re.findall(r'[\u3040-\u309f\u30a0-\u30ff]', value)
                    arabic_chars = re.findall(r'[\u0600-\u06ff]', value)
                    if chinese_chars or japanese_chars or arabic_chars:
                        logger.warning(
                            f"Invalid language in value '{value}': "
                            f"Chinese({len(chinese_chars)}), Japanese({len(japanese_chars)}), Arabic({len(arabic_chars)})"
                        )
                        return False
                return True
            elif isinstance(value, list):
                return all(check_value(item) for item in value)
            else:
                return True  # 숫자나 다른 타입은 통과

        for key, value in parsed_metadata.items():
            if not check_value(value):
                logger.warning(f"Language validation failed for key '{key}' with value: {value}")
                return False

        return True

    # -------- 메타데이터 생성 --------
    async def generate_metadata(self, text: str, existing_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """LLM을 사용해서 메타데이터 생성 (JsonOutputParser 사용)"""
        await self.ensure_llm_client()

        if not self.llm_client:
            logger.debug("LLM client not available, skipping metadata generation")
            return {}

        max_retries = 3
        for attempt in range(max_retries):
            try:
                from langchain_core.messages import HumanMessage, SystemMessage
                parser = JsonOutputParser(pydantic_object=DocumentMetadata)

                system_msg = SystemMessage(
                    content="당신은 문서 분석 전문가입니다. 주어진 텍스트를 분석해서 구조화된 메타데이터를 JSON 형식으로 정확하게 생성해주세요. 응답은 반드시 한국어 또는 영어만 사용해주세요."
                )
                human_msg = HumanMessage(content=self._create_metadata_prompt(text, existing_metadata))

                response = await self.llm_client.ainvoke([system_msg, human_msg])
                content = response.content.strip()

                # 1) 구조 파서 시도
                try:
                    parsed_result = parser.parse(content)
                    if not self._validate_parsed_metadata_language(parsed_result):
                        logger.warning(f"Parsed metadata language validation failed on attempt {attempt + 1}/{max_retries}")
                        if attempt < max_retries - 1:
                            continue
                        return {}
                    logger.info(f"Generated metadata using LLM: {list(parsed_result.keys())} (attempt {attempt + 1})")
                    return parsed_result

                # 2) Fallback JSON 수동 파싱 시도
                except Exception as parse_error:
                    logger.warning(f"JsonOutputParser failed: {parse_error}")

                    try:
                        # JSON 마크다운 블록 제거
                        cleaned = content
                        if cleaned.startswith("```json"):
                            cleaned = cleaned[7:]
                        if cleaned.endswith("```"):
                            cleaned = cleaned[:-3]
                        cleaned = cleaned.strip()

                        parsed_result = json.loads(cleaned)

                        validated_result = DocumentMetadata(**parsed_result)
                        validated_dict = validated_result.dict()

                        if not self._validate_parsed_metadata_language(validated_dict):
                            logger.warning(f"Fallback parsed metadata language validation failed on attempt {attempt + 1}/{max_retries}")
                            if attempt < max_retries - 1:
                                continue
                            return {}

                        logger.info(f"Generated metadata using fallback JSON parsing: {list(validated_dict.keys())} (attempt {attempt + 1})")
                        return validated_dict

                    except (json.JSONDecodeError, Exception) as fallback_error:
                        logger.error(f"Both JsonOutputParser and manual parsing failed: {fallback_error}")
                        logger.debug(f"Raw content: {content}")
                        if attempt < max_retries - 1:
                            continue
                        return {}

            except Exception as e:
                msg = str(e)
                # 모델 미존재/404 → 설정 재확인 후 즉시 재초기화
                if ("404" in msg and "does not exist" in msg) or ("NotFound" in msg):
                    provider, model, base_url, temperature, _ = self._compute_fingerprint()
                    logger.warning(
                        f"Model not found (attempt {attempt + 1}) against base_url={base_url}, model={model}, provider={provider} → reinit"
                    )
                    self._initialize_llm_client()
                    if attempt < max_retries - 1:
                        continue
                    return {}
                logger.error(f"Error generating metadata with LLM (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    continue
                return {}

        return {}

    async def generate_batch_metadata(
        self,
        texts: List[str],
        existing_metadatas: Optional[List[Dict[str, Any]]] = None,
        max_concurrent: int = 3,
    ) -> List[Dict[str, Any]]:
        """여러 텍스트에 대해 배치로 메타데이터 생성"""
        if not await self.is_enabled():
            logger.info("LLM metadata generation disabled")
            return [
                (existing_metadatas[i] if existing_metadatas and i < len(existing_metadatas) else {})
                for i in range(len(texts))
            ]

        # 동시 요청 수 제한
        semaphore = asyncio.Semaphore(max_concurrent)

        async def generate_single(i: int, text: str) -> Dict[str, Any]:
            async with semaphore:
                existing = existing_metadatas[i] if existing_metadatas and i < len(existing_metadatas) else None
                return await self.generate_metadata(text, existing)

        tasks = [generate_single(i, text) for i, text in enumerate(texts)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        processed_results: List[Dict[str, Any]] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error generating metadata for text {i}: {result}")
                existing = existing_metadatas[i] if existing_metadatas and i < len(existing_metadatas) else {}
                processed_results.append(existing)
            else:
                processed_results.append(result)

        return processed_results

class RAGService:

    _instance = None
    _instance_ref = None  # WeakReference로 인스턴스 추적

    def __new__(cls, vectordb_config, collection_config, openai_config=None):
        """싱글톤 패턴으로 인스턴스 생성 및 기존 인스턴스 완전 정리"""

        # 기존 인스턴스가 있으면 완전히 정리
        if cls._instance is not None:
            logger.info("Existing RAGService instance found. Performing complete cleanup...")
            try:
                # 기존 인스턴스 완전 정리
                cls._cleanup_existing_instance()
            except Exception as e:
                logger.error(f"Error during existing instance cleanup: {e}")

        # 새 인스턴스 생성
        logger.info("Creating new RAGService instance...")
        cls._instance = super(RAGService, cls).__new__(cls)
        cls._instance_ref = weakref.ref(cls._instance, cls._on_instance_deleted)

        return cls._instance

    @classmethod
    def _cleanup_existing_instance(cls):
        """기존 인스턴스 완전 정리"""
        if cls._instance is None:
            return

        try:
            # 임베딩 클라이언트 정리
            if hasattr(cls._instance, 'embeddings_client') and cls._instance.embeddings_client:
                try:
                    if hasattr(cls._instance.embeddings_client, 'cleanup'):
                        # 비동기 cleanup이 있으면 동기적으로 호출
                        import asyncio
                        try:
                            loop = asyncio.get_event_loop()
                            if loop.is_running():
                                # 이미 실행 중인 루프가 있으면 태스크로 스케줄링
                                asyncio.create_task(cls._instance.embeddings_client.cleanup())
                            else:
                                loop.run_until_complete(cls._instance.embeddings_client.cleanup())
                        except RuntimeError:
                            # 루프가 없거나 문제가 있으면 동기적으로 처리
                            pass
                    cls._instance.embeddings_client = None
                    logger.info("Embeddings client cleaned up")
                except Exception as e:
                    logger.warning(f"Error cleaning up embeddings client: {e}")

            # 벡터 매니저 정리
            if hasattr(cls._instance, 'vector_manager') and cls._instance.vector_manager:
                try:
                    if hasattr(cls._instance.vector_manager, 'cleanup'):
                        cls._instance.vector_manager.cleanup()
                    cls._instance.vector_manager = None
                    logger.info("Vector manager cleaned up")
                except Exception as e:
                    logger.warning(f"Error cleaning up vector manager: {e}")

            # 문서 프로세서 정리
            if hasattr(cls._instance, 'document_processor') and cls._instance.document_processor:
                try:
                    if hasattr(cls._instance.document_processor, 'cleanup'):
                        cls._instance.document_processor.cleanup()
                    cls._instance.document_processor = None
                    logger.info("Document processor cleaned up")
                except Exception as e:
                    logger.warning(f"Error cleaning up document processor: {e}")

            # LLM 메타데이터 생성기 정리
            if hasattr(cls._instance, 'metadata_generator'):
                cls._instance.metadata_generator = None
                logger.info("Metadata generator cleaned up")

            # 설정 참조 정리
            if hasattr(cls._instance, 'config'):
                cls._instance.config = None
            if hasattr(cls._instance, 'openai_config'):
                cls._instance.openai_config = None

            logger.info("Existing RAGService instance completely cleaned up")

        except Exception as e:
            logger.error(f"Error during RAGService cleanup: {e}")
        finally:
            # 가비지 컬렉션 강제 실행
            cls._instance = None
            cls._instance_ref = None
            gc.collect()
            logger.info("Garbage collection completed after RAGService cleanup")

    @classmethod
    def _on_instance_deleted(cls, ref):
        """인스턴스가 삭제될 때 호출되는 콜백"""
        logger.info("RAGService instance has been garbage collected")
        cls._instance = None
        cls._instance_ref = None

    def __init__(self, vectordb_config, collection_config, openai_config=None):
        """RAGService 초기화

        Args:
            vectordb_config: 벡터 DB 설정 객체
            collection_config: 컬렉션 설정 객체
            openai_config: OpenAI 설정 객체 (하위 호환성)
        """
        # 이미 초기화된 인스턴스인지 확인
        if hasattr(self, '_initialized') and self._initialized:
            logger.info("RAGService instance already initialized, skipping...")
            return

        logger.info("Initializing RAGService components...")

        self.config = vectordb_config
        self.openai_config = openai_config

        # 컴포넌트 초기화
        self.document_processor = DocumentProcessor(collection_config)
        self.document_processor.test()

        self.vector_manager = VectorManager(vectordb_config)
        self.embeddings_client = None

        # LLM 메타데이터 생성기 초기화
        self.metadata_generator = LLMMetadataGenerator(collection_config)

        self._initialize_embeddings()

        # 초기화 완료 마킹
        self._initialized = True
        logger.info("RAGService initialization completed")

    def cleanup(self):
        """현재 인스턴스 정리 (외부에서 호출 가능)"""
        logger.info("Manual RAGService cleanup requested")
        self._cleanup_existing_instance()

    @classmethod
    def get_instance(cls):
        """현재 인스턴스 반환 (있는 경우)"""
        return cls._instance

    @classmethod
    def is_initialized(cls):
        """인스턴스가 초기화되어 있는지 확인"""
        return cls._instance is not None and hasattr(cls._instance, '_initialized') and cls._instance._initialized

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

                        # dimension이 유효한 정수일 때만 업데이트 시도
                        if isinstance(dimension, int) and dimension > 0:
                            logger.info(f"Embedding client initialized successfully: {provider}, dimension: {dimension}")

                            # 설정에서 AUTO_DETECT_EMBEDDING_DIM이 True면 차원 수 업데이트
                            if self.config.AUTO_DETECT_EMBEDDING_DIM.value:
                                old_dimension = self.config.VECTOR_DIMENSION.value
                                if old_dimension != dimension:
                                    logger.info(f"Updating vector dimension from {old_dimension} to {dimension}")
                                    # 메모리와 DB에 모두 저장하여 _config_refresh 시 변경이 유지되도록 함
                                    try:
                                        self.config.VECTOR_DIMENSION.value = dimension
                                        try:
                                            self.config.VECTOR_DIMENSION.save()
                                        except Exception as save_e:
                                            logger.warning(f"Failed to persist VECTOR_DIMENSION to DB: {save_e}")

                                        # reinitialize vector manager with new persisted value
                                        self._reinitialize_vector_manager()
                                    except Exception as e:
                                        logger.warning(f"Failed while updating VECTOR_DIMENSION: {e}")

                            return
                        else:
                            logger.info("Embedding dimension unknown at init time; skipping VECTOR_DIMENSION update")
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

    async def process_document(self, user_id, app_db, file_path: str, collection_name: str,
                                chunk_size: int = 1000, chunk_overlap: int = 200,
                                metadata: Dict[str, Any] = None,
                                use_llm_metadata: bool = True,
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
            # processed_chunks 변수가 이후 조건에서 사용되므로 안전하게 초기화
            processed_chunks = []
            # 파일 확장자 추출 및 검증
            is_valid, file_extension = self.document_processor.validate_file_format(file_path)
            if not is_valid:
                raise ValueError(f"Unsupported file type: {file_extension}")

            # 텍스트 추출
            logger.info(f"Extracting text from {file_path}")
            text = await self.document_processor.extract_text_from_file(file_path, file_extension)

            if not text.strip():
                raise ValueError("No text content found in the document")

            # 텍스트 청킹과 메타데이터 생성 (document_processor에 해당 메서드가 없으면 폴백)
            logger.info(f"Chunking text with size {chunk_size} and overlap {chunk_overlap}")
            try:
                chunks_with_metadata = self.document_processor.chunk_text_with_metadata(
                    text, file_extension, chunk_size, chunk_overlap
                )
            except AttributeError:
                # 서버 재시작 없이 코드가 변경되어 현재 인스턴스에 메서드가 없을 수 있음.
                logger.warning("DocumentProcessor.chunk_text_with_metadata not found on current instance. Attempting to reinitialize DocumentProcessor and retry.")
                reinitialized = False
                try:
                    # 모듈이 디스크에서 업데이트되었을 수 있으므로 런타임에서 모듈을 리로드한 뒤 인스턴스 생성
                    import importlib
                    import service.retrieval.document_processor as dp_mod

                    importlib.reload(dp_mod)
                    DocumentProcessorClass = getattr(dp_mod, 'DocumentProcessor', None)

                    if DocumentProcessorClass is None:
                        raise RuntimeError('DocumentProcessor class not found after reload')

                    collection_config = getattr(self.document_processor, 'collection_config', None)
                    self.document_processor = DocumentProcessorClass(collection_config)
                    logger.info("Reinitialized DocumentProcessor instance from reloaded module; retrying chunk_text_with_metadata")
                    chunks_with_metadata = self.document_processor.chunk_text_with_metadata(
                        text, file_extension, chunk_size, chunk_overlap
                    )
                    reinitialized = True
                except Exception as e:
                    logger.warning(f"Reinitialization failed or method still unavailable: {e}. Trying explicit file-load fallback.")

                    # 시나리오: package와 top-level module명이 충돌하여 패키지 내부 모듈이 사용될 수 있음.
                    # 이 경우 프로젝트의 최상위 document_processor.py 파일을 명시적 경로로 로드해 시도해본다.
                    try:
                        import importlib.util, sys

                        # use top-level Path imported at module scope
                        base_path = Path(__file__).resolve().parents[3]
                        target_path = base_path / 'service' / 'retrieval' / 'document_processor.py'

                        if target_path.exists():
                            spec = importlib.util.spec_from_file_location("plateerag_document_processor_file", str(target_path))
                            module = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(module)
                            DocumentProcessorClass = getattr(module, 'DocumentProcessor', None)
                            if DocumentProcessorClass is not None:
                                collection_config = getattr(self.document_processor, 'collection_config', None)
                                self.document_processor = DocumentProcessorClass(collection_config)
                                logger.info("Loaded DocumentProcessor class from file path and reinitialized instance; retrying chunk_text_with_metadata")
                                chunks_with_metadata = self.document_processor.chunk_text_with_metadata(
                                    text, file_extension, chunk_size, chunk_overlap
                                )
                                reinitialized = True
                            else:
                                logger.warning("DocumentProcessor not found in file-loaded module")
                        else:
                            logger.warning(f"Top-level document_processor.py not found at {target_path}")
                    except Exception as e2:
                        logger.warning(f"File-load fallback failed: {e2}. Falling back to chunk_text + metadata builder.")

                if not reinitialized:
                    chunks = self.document_processor.chunk_text(text, chunk_size, chunk_overlap)

                    # build line starts using document_processor helper if available
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

                vectordb_chunk_meta = VectorDBChunkMeta(
                    user_id=user_id,
                    collection_name=collection_name,
                    file_name=file_name,
                    chunk_id=point['id'],
                    chunk_text=(payload.get("chunk_text")[:500] + "..." if len(payload.get("chunk_text")) > 500 else payload.get("chunk_text")),
                    chunk_index=payload.get("chunk_index"),
                    total_chunks=payload.get("total_chunks"),
                    chunk_size=payload.get("chunk_size"),
                    summary=payload.get("summary"),
                    keywords=safe_list_to_string(payload.get("keywords")),
                    topics=safe_list_to_string(payload.get("topics")),
                    entities=safe_list_to_string(payload.get("entities")),
                    sentiment=payload.get("sentiment"),
                    document_type=payload.get("document_type"),
                    language=payload.get("language"),
                    complexity_level=payload.get("complexity_level"),
                    main_concepts=safe_list_to_string(payload.get("main_concepts")),
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

                            # 타임아웃 설정
                            try:
                                RERANK_TIMEOUT = getattr(self.config, 'RERANK_TIMEOUT', None)
                                timeout_sec = int(RERANK_TIMEOUT.value) if RERANK_TIMEOUT and hasattr(RERANK_TIMEOUT, 'value') else 10
                            except Exception:
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

            # 문서 기본 정보 추출
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

    async def get_llm_metadata_status(self) -> Dict[str, Any]:
        """LLM 메타데이터 생성기 상태 조회"""
        try:
            if not self.metadata_generator:
                return {
                    "status": "not_initialized",
                    "enabled": False,
                    "error": "Metadata generator not initialized"
                }

            # ✅ collection_config에서 안전하게 값 가져오기
            provider = self.metadata_generator._get_config_value('IMAGE_TEXT_MODEL_PROVIDER', 'no_model')
            model = self.metadata_generator._get_config_value('IMAGE_TEXT_MODEL_NAME', 'unknown')
            base_url = self.metadata_generator._get_config_value('IMAGE_TEXT_BASE_URL', '')
            temperature = self.metadata_generator._get_config_value('IMAGE_TEXT_TEMPERATURE', 0.1)

            is_enabled = await self.metadata_generator.is_enabled()

            return {
                "status": "initialized",
                "enabled": is_enabled,
                "provider": str(provider),
                "model": str(model),
                "base_url": str(base_url),
                "temperature": float(temperature),
                "client_initialized": bool(self.metadata_generator.llm_client)
            }
        except Exception as e:
            return {
                "status": "error",
                "enabled": False,
                "error": str(e)
            }

    def get_config(self) -> Dict[str, Any]:
        """현재 RAG 서비스의 모든 설정 반환"""
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
                        "document_processor": bool(self.document_processor),
                        "metadata_generator": bool(self.metadata_generator)
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

            # ✅ LLM 메타데이터 정보 추가 (전달받은 설정만 사용)
            try:
                if self.metadata_generator:
                    llm_config = self.metadata_generator.config
                    config_info["llm_metadata"] = {
                        "provider": llm_config.get('provider', 'no_model'),
                        "model": llm_config.get('model', 'unknown'),
                        "base_url": llm_config.get('base_url', ''),
                        "temperature": llm_config.get('temperature', 0.1),
                        "client_initialized": bool(self.metadata_generator.llm_client)
                    }
                else:
                    config_info["llm_metadata"] = {
                        "provider": "no_model",
                        "error": "Metadata generator not initialized"
                    }
            except Exception as e:
                config_info["llm_metadata"] = {
                    "error": str(e),
                    "provider": "unknown"
                }

            return config_info

        except Exception as e:
            logger.error(f"Failed to get RAG service config: {e}")
            return {
                "error": str(e),
                "basic_info": {
                    "provider": self.config.EMBEDDING_PROVIDER.value if self.config else "unknown",
                    "vector_db_connected": self.vector_manager.is_connected() if self.vector_manager else False,
                    "embedding_client_initialized": bool(self.embeddings_client),
                    "metadata_generator_initialized": bool(self.metadata_generator)
                }
            }

    # DEPRECATED: 이 메서드는 더 이상 사용되지 않습니다. document_processor.chunk_text_with_metadata를 사용하세요.
    def _extract_page_mapping(self, text: str, file_extension: str) -> List[Dict[str, Any]]:
        """텍스트에서 페이지 정보 추출

        Args:
            text: 원본 텍스트
            file_extension: 파일 확장자

        Returns:
            페이지 정보 리스트 [{"page_num": 1, "start_pos": 0, "end_pos": 100}, ...]
        """
        try:
            page_mapping = []

            # PDF, PPT, DOCX에서 페이지 구분자 패턴 찾기
            if file_extension in ['pdf', 'ppt', 'pptx', 'docx', 'doc']:
                import re

                # 페이지 구분자 패턴들
                patterns = [
                    r'=== 페이지 (\d+) ===',  # PDF PyPDF2 fallback, DOCX 기본 텍스트 추출
                    r'=== 페이지 (\d+) \(OCR\) ===',  # PDF OCR
                    r'=== 슬라이드 (\d+) ===',  # PPT 기본
                    r'=== 슬라이드 (\d+) \(OCR\) ===',  # PPT OCR
                ]

                found_pages = False
                for pattern in patterns:
                    matches = list(re.finditer(pattern, text))
                    if matches:
                        logger.info(f"Found {len(matches)} page markers with pattern: {pattern}")

                        for i, match in enumerate(matches):
                            page_num = int(match.group(1))
                            start_pos = match.end()  # 페이지 제목 다음부터

                            # 다음 페이지의 시작 위치 또는 텍스트 끝
                            if i + 1 < len(matches):
                                end_pos = matches[i + 1].start()
                            else:
                                end_pos = len(text)

                            page_mapping.append({
                                "page_num": page_num,
                                "start_pos": start_pos,
                                "end_pos": end_pos
                            })

                        # 페이지 번호 순으로 정렬
                        page_mapping.sort(key=lambda x: x["page_num"])
                        found_pages = True
                        break

                if not found_pages:
                    # DOCX에서 OCR을 통해 페이지가 구분되었는지 확인
                    if file_extension in ['docx', 'doc']:
                        ocr_page_pattern = r'\n=== 페이지 (\d+) \(OCR\) ===\n'
                        ocr_matches = list(re.finditer(ocr_page_pattern, text))
                        if ocr_matches:
                            logger.info(f"Found {len(ocr_matches)} DOCX OCR page markers")
                            for i, match in enumerate(ocr_matches):
                                page_num = int(match.group(1))
                                start_pos = match.end()

                                if i + 1 < len(ocr_matches):
                                    end_pos = ocr_matches[i + 1].start()
                                else:
                                    end_pos = len(text)

                                page_mapping.append({
                                    "page_num": page_num,
                                    "start_pos": start_pos,
                                    "end_pos": end_pos
                                })
                            page_mapping.sort(key=lambda x: x["page_num"])
                            found_pages = True
                        else:
                            # OCR 페이지 구분자도 없으면 텍스트 길이 기준으로 가상 페이지 생성
                            # DOCX의 경우 대략 1500자당 1페이지로 추정
                            chars_per_page = 1500
                            text_length = len(text)

                            if text_length > chars_per_page:
                                estimated_pages = (text_length + chars_per_page - 1) // chars_per_page
                                logger.info(f"Creating {estimated_pages} virtual pages for DOCX based on text length ({text_length} chars)")

                                for page_num in range(1, estimated_pages + 1):
                                    start_pos = (page_num - 1) * chars_per_page
                                    end_pos = min(page_num * chars_per_page, text_length)

                                    page_mapping.append({
                                        "page_num": page_num,
                                        "start_pos": start_pos,
                                        "end_pos": end_pos
                                    })
                                found_pages = True

                    if not found_pages:
                        logger.info("No page markers found, treating as single page document")
                        page_mapping = [{"page_num": 1, "start_pos": 0, "end_pos": len(text)}]

            elif file_extension in ['xlsx', 'xls']:
                # Excel: 시트별로 페이지 구분
                import re
                sheet_pattern = r'\n=== 시트: ([^=]+) ===\n'
                matches = list(re.finditer(sheet_pattern, text))

                if matches:
                    logger.info(f"Found {len(matches)} Excel sheets")
                    for i, match in enumerate(matches):
                        sheet_name = match.group(1)
                        page_num = i + 1
                        start_pos = match.end()

                        # 다음 시트의 시작 위치 또는 텍스트 끝
                        if i + 1 < len(matches):
                            end_pos = matches[i + 1].start()
                        else:
                            end_pos = len(text)

                        page_mapping.append({
                            "page_num": page_num,
                            "start_pos": start_pos,
                            "end_pos": end_pos,
                            "sheet_name": sheet_name
                        })
                else:
                    page_mapping = [{"page_num": 1, "start_pos": 0, "end_pos": len(text)}]

            else:
                # 텍스트 파일이나 기타 형식: 줄 수 기준으로 가상 페이지 생성 (1000줄당 1페이지)
                lines = text.split('\n')
                lines_per_page = 1000
                total_lines = len(lines)

                if total_lines <= lines_per_page:
                    page_mapping = [{"page_num": 1, "start_pos": 0, "end_pos": len(text)}]
                else:
                    current_pos = 0
                    page_num = 1

                    for i in range(0, total_lines, lines_per_page):
                        start_line = i
                        end_line = min(i + lines_per_page, total_lines)

                        # 라인 범위를 텍스트 위치로 변환
                        if start_line == 0:
                            start_pos = 0
                        else:
                            start_pos = len('\n'.join(lines[:start_line])) + 1  # +1 for newline

                        if end_line >= total_lines:
                            end_pos = len(text)
                        else:
                            end_pos = len('\n'.join(lines[:end_line]))

                        page_mapping.append({
                            "page_num": page_num,
                            "start_pos": start_pos,
                            "end_pos": end_pos,
                            "start_line": start_line + 1,
                            "end_line": end_line
                        })
                        page_num += 1

                    logger.info(f"Created {len(page_mapping)} virtual pages for text file ({total_lines} lines)")

            return page_mapping

        except Exception as e:
            logger.warning(f"Error extracting page mapping: {e}")
            return [{"page_num": 1, "start_pos": 0, "end_pos": len(text)}]

    # DEPRECATED: 이 메서드는 더 이상 사용되지 않습니다. document_processor.chunk_text_with_metadata를 사용하세요.
    def _get_page_number_for_chunk(self, chunk: str, full_text: str, chunk_pos: int, page_mapping: List[Dict[str, Any]]) -> int:
        """청크의 페이지 번호 계산

        Args:
            chunk: 청크 텍스트
            full_text: 전체 텍스트
            chunk_pos: 청크의 전체 텍스트 내 위치 (-1이면 찾기 실패)
            page_mapping: 페이지 매핑 정보

        Returns:
            페이지 번호
        """
        try:
            if not page_mapping:
                return 1

            # 단일 페이지인 경우
            if len(page_mapping) == 1:
                return 1

            if chunk_pos == -1:
                # 위치를 찾지 못한 경우 청크 내용으로 추정
                # 여러 가지 방법으로 위치 찾기 시도
                search_attempts = [
                    chunk[:100] if len(chunk) > 100 else chunk,  # 첫 100자
                    chunk[:50] if len(chunk) > 50 else chunk,    # 첫 50자
                    chunk.split('\n')[0] if '\n' in chunk else chunk[:30],  # 첫 줄 또는 첫 30자
                ]

                for attempt in search_attempts:
                    if attempt.strip():
                        chunk_pos = full_text.find(attempt.strip())
                        if chunk_pos != -1:
                            break

                if chunk_pos == -1:
                    logger.warning("Could not determine chunk position, defaulting to page 1")
                    return 1

            # 청크의 시작, 중간, 끝 위치를 모두 고려
            chunk_start = chunk_pos
            chunk_middle = chunk_pos + len(chunk) // 2
            chunk_end = chunk_pos + len(chunk)

            # 세 위치 중 어느 하나라도 페이지 범위에 속하면 해당 페이지로 결정
            for page_info in page_mapping:
                start_pos = page_info["start_pos"]
                end_pos = page_info["end_pos"]

                # 청크가 페이지 범위와 겹치는지 확인
                if (start_pos <= chunk_start < end_pos or
                    start_pos <= chunk_middle < end_pos or
                    start_pos <= chunk_end <= end_pos or
                    (chunk_start <= start_pos and chunk_end >= end_pos)):  # 청크가 페이지를 완전히 포함하는 경우
                    return page_info["page_num"]

            # 위치 기반으로 찾지 못한 경우, 가장 가까운 페이지 찾기
            min_distance = float('inf')
            closest_page = 1

            for page_info in page_mapping:
                start_pos = page_info["start_pos"]
                end_pos = page_info["end_pos"]

                # 청크 중간점과 페이지 중간점의 거리 계산
                page_middle = (start_pos + end_pos) // 2
                distance = abs(chunk_middle - page_middle)

                if distance < min_distance:
                    min_distance = distance
                    closest_page = page_info["page_num"]

            logger.debug(f"Assigned chunk at position {chunk_pos} to closest page {closest_page}")
            return closest_page

        except Exception as e:
            logger.warning(f"Error calculating page number for chunk: {e}")
            return 1

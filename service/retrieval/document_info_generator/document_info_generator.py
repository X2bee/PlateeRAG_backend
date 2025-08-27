import re
import json
import asyncio
import logging

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser

logger = logging.getLogger("document-info-generator")

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

class DocumentInfoGenerator:
    """LLM 기반 메타데이터 생성 클래스 (설정 변경 자동 감지/재초기화 버전)"""
    def __init__(self, config_composer = None):
        self.config_composer = config_composer
        self.document_processor_config = self._get_current_document_processor_config()
        self.llm_client = None

    def _get_current_document_processor_config(self) -> Dict[str, Any]:
        if self.config_composer:
            document_processor_config = self.config_composer.get_config_by_category_name("document-processor")
            provider = document_processor_config.DOCUMENT_PROCESSOR_IMAGE_TEXT_MODEL_PROVIDER.value

            if provider == "openai":
                config = {
                    'provider': str(document_processor_config.DOCUMENT_PROCESSOR_IMAGE_TEXT_MODEL_PROVIDER.value).lower(),
                    'base_url': str(document_processor_config.DOCUMENT_PROCESSOR_OPENAI_IMAGE_TEXT_BASE_URL.value),
                    'api_key': str(document_processor_config.DOCUMENT_PROCESSOR_OPENAI_IMAGE_TEXT_API_KEY.value),
                    'model': str(document_processor_config.DOCUMENT_PROCESSOR_OPENAI_IMAGE_TEXT_MODEL_NAME.value),
                    'temperature': float(document_processor_config.DOCUMENT_PROCESSOR_IMAGE_TEXT_TEMPERATURE.value),
                }

            elif provider == "vllm":
                config = {
                    'provider': str(document_processor_config.DOCUMENT_PROCESSOR_IMAGE_TEXT_MODEL_PROVIDER.value).lower(),
                    'base_url': str(document_processor_config.DOCUMENT_PROCESSOR_VLLM_IMAGE_TEXT_BASE_URL.value),
                    'api_key': str(document_processor_config.DOCUMENT_PROCESSOR_VLLM_IMAGE_TEXT_API_KEY.value),
                    'model': str(document_processor_config.DOCUMENT_PROCESSOR_VLLM_IMAGE_TEXT_MODEL_NAME.value),
                    'temperature': float(document_processor_config.DOCUMENT_PROCESSOR_IMAGE_TEXT_TEMPERATURE.value),
                }

            elif provider == "no_model":
                config = {
                    'provider': str(document_processor_config.DOCUMENT_PROCESSOR_IMAGE_TEXT_MODEL_PROVIDER.value).lower(),
                    'base_url': "",
                    'api_key': "",
                    'model': "",
                    'temperature': 0,
                }

            else:
                raise ValueError(f"Unsupported IMAGE_TEXT_MODEL_PROVIDER: {provider}")
        else:
            raise ValueError("Config composer not provided")

        return config

    def _config_changed(self, new_config: Dict[str, Any]) -> bool:
        """설정이 변경되었는지 안전하게 비교"""
        if self.document_processor_config is None:
            return True

        # 키가 다르면 변경됨
        if set(self.document_processor_config.keys()) != set(new_config.keys()):
            return True

        # 각 값을 비교 (float는 약간의 오차 허용)
        for key in self.document_processor_config:
            old_val = self.document_processor_config[key]
            new_val = new_config[key]

            if isinstance(old_val, float) and isinstance(new_val, float):
                if abs(old_val - new_val) > 1e-6:  # 부동소수점 오차 허용
                    return True
            else:
                if old_val != new_val:
                    return True

        return False

    # -------- 클라이언트 초기화/활성화 --------
    def _initialize_llm_client(self):
        """LLM 클라이언트 초기화 (설정 변화가 있으면 항상 새로 초기화)"""
        try:
            cfg = self._get_current_document_processor_config()

            provider = cfg.get('provider', 'unknown')
            base_url = cfg.get('base_url', 'unknown')
            api_key = cfg.get('api_key', '')
            model = cfg.get('model', 'unknown')
            temperature = cfg.get('temperature', 'unknown')

            if provider == "no_model":
                logger.info("No LLM model configured - metadata generation disabled")
                self.llm_client = None
                return

            logger.info(f"Initializing LLM client with provider={provider}, model={model}, base_url={base_url}")

            if provider in ("openai", "vllm"):
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
                return

            logger.info(f"LLM client initialized successfully: {provider} / {model} @ {base_url}")

        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
            self.llm_client = None

    async def is_enabled(self) -> bool:
        """메타데이터 생성이 활성화되어 있는지 확인"""
        try:
            provider = str(self.config_composer.get_config_by_name("DOCUMENT_PROCESSOR_IMAGE_TEXT_MODEL_PROVIDER").value)
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

        current_fp = self._get_current_document_processor_config()
        if (self.llm_client is None) or self._config_changed(current_fp):
            logger.info("LLM client missing or config changed → reinitializing")
            self.document_processor_config = current_fp  # 설정 업데이트
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

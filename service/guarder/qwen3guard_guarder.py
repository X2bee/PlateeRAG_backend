"""
Qwen3Guard Guarder 클라이언트 (transformers 사용)
"""

import asyncio
from typing import Dict, Any, List
import logging
import re
import os
from service.guarder.base_guarder import BaseGuarder

logger = logging.getLogger("guarder.qwen3guard")

class Qwen3GuardGuarder(BaseGuarder):
    """Qwen3Guard transformers Guarder 클라이언트"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_name = config.get("model_name", "Qwen/Qwen3Guard-Gen-0.6B")
        self.model_device = config.get("model_device", "cpu")
        self.api_key = config.get("api_key", "")
        self.tokenizer = None
        self.model = None

        self._initialize_model()

    def _initialize_model(self):
        """모델 초기화"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            # API key가 있으면 설정
            if self.api_key:
                os.environ["HUGGINGFACE_HUB_TOKEN"] = self.api_key

            logger.info("Loading Qwen3Guard model: %s", self.model_name)

            try:
                is_gpu_device = (
                    self.model_device == 'gpu' or
                    self.model_device == 'cuda'
                )
                logger.info("Using device: %s (GPU: %s)", self.model_device, is_gpu_device)
                device_map = 'auto' if is_gpu_device else None

                # 토크나이저와 모델 로드
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype="auto",
                    device_map=device_map
                )

                # CPU 사용 시 명시적으로 디바이스 이동
                if not is_gpu_device:
                    self.model.to('cpu')

                logger.info("Guarder Model loaded on ===%s=== device",
                           'cuda' if is_gpu_device else 'cpu')
                logger.info("Qwen3Guard model loaded successfully: %s", self.model_name)

            except (ImportError, OSError, RuntimeError) as model_error:
                raise RuntimeError(f"Failed to load Qwen3Guard model: {model_error}") from model_error

        except ImportError as import_error:
            logger.error("transformers package not installed. Run: pip install transformers")
            logger.error("Import error details: %s", import_error)
            self.tokenizer = None
            self.model = None
        except (RuntimeError, OSError) as e:
            logger.error("Failed to initialize Qwen3Guard model: %s", e)
            self.tokenizer = None
            self.model = None

    def _extract_label_and_categories(self, content: str) -> tuple:
        """
        모델 출력에서 안전성 라벨과 카테고리를 추출

        Args:
            content: 모델의 원시 출력

        Returns:
            (label, categories) 튜플
        """
        safe_pattern = r"Safety: (Safe|Unsafe|Controversial)"
        category_pattern = r"(Violent|Non-violent Illegal Acts|Sexual Content or Sexual Acts|PII|Suicide & Self-Harm|Unethical Acts|Politically Sensitive Topics|Copyright Violation|Jailbreak|None)"

        safe_label_match = re.search(safe_pattern, content)
        label = safe_label_match.group(1) if safe_label_match else "Unknown"

        categories = re.findall(category_pattern, content)

        return label, categories

    async def moderate_text(self, text: str) -> Dict[str, Any]:
        """텍스트 내용을 검사하여 안전성을 판단"""
        if not self.model or not self.tokenizer:
            raise ValueError("Qwen3Guard model not initialized")

        try:
            # CPU 집약적인 작업을 별도 스레드에서 실행
            moderation_result = await asyncio.to_thread(
                self._moderate_text_sync, text
            )

            logger.info("Text moderation completed for text: %s...", text[:50])
            return moderation_result

        except (ValueError, ImportError, RuntimeError) as e:
            logger.error("Failed to moderate text: %s", e)
            raise

    def _moderate_text_sync(self, text: str) -> Dict[str, Any]:
        """텍스트를 검사 (동기 함수)"""
        try:
            # 프롬프트 모더레이션을 위한 메시지 형식
            messages = [
                {"role": "user", "content": text}
            ]

            # 채팅 템플릿 적용
            formatted_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False
            )

            # 모델 입력 준비
            model_inputs = self.tokenizer([formatted_text], return_tensors="pt").to(self.model.device)

            # 텍스트 생성
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=128
            )

            # 입력 부분을 제외한 생성된 부분만 추출
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

            # 토큰을 텍스트로 디코드
            raw_content = self.tokenizer.decode(output_ids, skip_special_tokens=True)

            # 라벨과 카테고리 추출
            label, categories = self._extract_label_and_categories(raw_content)

            # 안전성 판단
            is_safe, filtered_categories = self._parse_safety_categories(categories)

            # 라벨 기반 안전성 재검증
            if label == "Unsafe":
                is_safe = False
            elif label == "Safe":
                is_safe = True
            # Controversial의 경우 카테고리 기반 판단 유지

            # 신뢰도 계산 (간단한 휴리스틱)
            confidence = 0.9 if label in ["Safe", "Unsafe"] else 0.7
            if label == "Unknown":
                confidence = 0.3

            result = {
                "safe": is_safe,
                "label": label,
                "categories": filtered_categories,
                "confidence": confidence,
                "raw_response": raw_content.strip()
            }

            logger.debug("Moderation result: %s", result)
            return result

        except Exception as e:
            logger.error("Error in text moderation: %s", e)
            raise

    async def is_available(self) -> bool:
        """Qwen3Guard 서비스 사용 가능성 확인"""
        if not self.model or not self.tokenizer:
            return False

        try:
            # 간단한 테스트 수행
            test_result = await self.moderate_text("Hello, world!")
            return isinstance(test_result, dict) and "safe" in test_result

        except (RuntimeError, AttributeError, ValueError) as e:
            logger.warning("Qwen3Guard service not available: %s", e)
            return False

    def get_provider_info(self) -> Dict[str, Any]:
        """Qwen3Guard 제공자 정보 반환"""
        return {
            "provider": "qwen3guard",
            "model": self.model_name,
            "api_key_configured": bool(self.api_key),
            "available": bool(self.model and self.tokenizer)
        }

    async def cleanup(self):
        """Qwen3Guard 모델 리소스 정리"""
        logger.info("Cleaning up Qwen3Guard client: %s", self.model_name)

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

                # 토크나이저 객체 정리
                del self.tokenizer
                self.tokenizer = None

                logger.info("Qwen3Guard model cleanup completed")

            except (RuntimeError, AttributeError) as e:
                logger.warning("Error during Qwen3Guard model cleanup: %s", e)

        # 부모 클래스 cleanup 호출
        await super().cleanup()

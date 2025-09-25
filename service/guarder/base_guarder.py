"""
Guarder 클라이언트의 기본 추상 클래스
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
import logging

logger = logging.getLogger("guarder")

class BaseGuarder(ABC):
    """Guarder 클라이언트의 기본 추상 클래스"""

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Guarder 설정 딕셔너리
        """
        self.config = config
        self.provider_name = self.__class__.__name__.replace("Guarder", "").lower()

    @abstractmethod
    async def moderate_text(self, text: str) -> Dict[str, Any]:
        """
        텍스트 내용을 검사하여 안전성을 판단

        Args:
            text: 검사할 텍스트

        Returns:
            검사 결과 딕셔너리
            {
                "safe": bool,           # 안전 여부 (True: Safe, False: Unsafe)
                "label": str,           # 안전성 라벨 ("Safe", "Unsafe", "Controversial")
                "categories": List[str], # 위험 카테고리 목록
                "confidence": float,    # 신뢰도 (0.0-1.0)
                "raw_response": str     # 원본 응답
            }
        """
        raise NotImplementedError

    @abstractmethod
    async def is_available(self) -> bool:
        """
        Guarder 서비스가 사용 가능한지 확인

        Returns:
            서비스 사용 가능 여부
        """
        raise NotImplementedError

    def get_provider_info(self) -> Dict[str, Any]:
        """
        Guarder 제공자 정보 반환

        Returns:
            제공자 정보 딕셔너리
        """
        return {
            "provider": self.provider_name,
            "model": self.config.get("model", "unknown"),
            "available": False  # 서브클래스에서 오버라이드
        }

    async def cleanup(self):
        """
        Guarder 클라이언트 리소스 정리
        서브클래스에서 필요시 오버라이드
        """
        logger.info("Cleaning up %s Guarder client", self.provider_name)

    def _parse_safety_categories(self, categories: List[str]) -> Tuple[bool, List[str]]:
        """
        카테고리 목록을 기반으로 안전성 판단

        Args:
            categories: 검출된 카테고리 목록

        Returns:
            (is_safe, filtered_categories)
        """
        # "None" 카테고리는 제외하고 실제 위험 카테고리만 필터링
        filtered_categories = [cat for cat in categories if cat != "None"]

        # 위험 카테고리가 없으면 안전
        is_safe = len(filtered_categories) == 0

        return is_safe, filtered_categories

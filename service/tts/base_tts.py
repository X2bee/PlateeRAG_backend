"""
TTS 클라이언트의 기본 추상 클래스
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger("tts")

class BaseTTS(ABC):
    """TTS 클라이언트의 기본 추상 클래스"""

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: TTS 설정 딕셔너리
        """
        self.config = config
        self.provider_name = self.__class__.__name__.replace("TTS", "").lower()

    @abstractmethod
    async def generate_speech(
        self,
        text: str,
        speaker: Optional[str] = None,
        language: str = "ko",
        output_format: str = "wav"
    ) -> bytes:
        """
        텍스트를 음성으로 변환

        Args:
            text: 변환할 텍스트
            speaker: 화자 설정 (선택사항)
            language: 언어 코드 (기본값: "ko")
            output_format: 출력 오디오 형식 (wav, mp3 등)

        Returns:
            생성된 오디오 데이터 (bytes)
        """
        raise NotImplementedError

    @abstractmethod
    async def is_available(self) -> bool:
        """
        TTS 서비스가 사용 가능한지 확인

        Returns:
            서비스 사용 가능 여부
        """
        raise NotImplementedError

    def get_provider_info(self) -> Dict[str, Any]:
        """
        TTS 제공자 정보 반환

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
        TTS 클라이언트 리소스 정리
        서브클래스에서 필요시 오버라이드
        """
        logger.info("Cleaning up %s TTS client", self.provider_name)

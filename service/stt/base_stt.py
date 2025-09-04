"""
STT 클라이언트의 기본 추상 클래스
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Union
import logging

logger = logging.getLogger("stt")

class BaseSTT(ABC):
    """STT 클라이언트의 기본 추상 클래스"""

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: STT 설정 딕셔너리
        """
        self.config = config
        self.provider_name = self.__class__.__name__.replace("STT", "").lower()

    @abstractmethod
    async def transcribe_audio(self, audio_data: Union[bytes, str], audio_format: str = "wav") -> str:
        """
        오디오 데이터를 텍스트로 변환

        Args:
            audio_data: 오디오 데이터 (bytes) 또는 파일 경로 (str)
            audio_format: 오디오 형식 (wav, mp3, flac 등)

        Returns:
            변환된 텍스트
        """
        pass

    @abstractmethod
    async def is_available(self) -> bool:
        """
        STT 서비스가 사용 가능한지 확인

        Returns:
            서비스 사용 가능 여부
        """
        pass

    def get_provider_info(self) -> Dict[str, Any]:
        """
        STT 제공자 정보 반환

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
        STT 클라이언트 리소스 정리
        서브클래스에서 필요시 오버라이드
        """
        logger.info("Cleaning up %s STT client", self.provider_name)

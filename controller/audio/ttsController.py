"""
TTS API 라우터 및 컨트롤러 통합
텍스트를 음성으로 변환하는 REST API 엔드포인트 제공
"""

import logging
from typing import Optional
from fastapi.responses import JSONResponse
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from service.tts.tts_factory import TTSFactory
from controller.helper.singletonHelper import get_tts_service, get_config_composer
import io
from service.tts.tts_factory import TTSFactory
from controller.admin.adminBaseController import validate_superuser

logger = logging.getLogger("controller.tts")

# TTS 요청 모델
class TTSRequest(BaseModel):
    """TTS 요청 모델"""
    text: str
    speaker: Optional[str] = None
    output_format: str = "wav"
    # 감정 값들 (Happiness, Sadness, Disgust, Fear, Surprise, Anger, Other, Neutral)
    happiness: float = 0.3077
    sadness: float = 0.0256
    disgust: float = 0.0256
    fear: float = 0.0256
    surprise: float = 0.0256
    anger: float = 0.0256
    other: float = 0.2564
    neutral: float = 0.3077

router = APIRouter(
    prefix="/tts",
    tags=["TTS (Text-to-Speech)"],
    responses={
        500: {"description": "Internal server error"},
        503: {"description": "Service unavailable"}
    }
)

class TTSController:
    """TTS 컨트롤러"""

    def __init__(self, config_composer):
        self.config_composer = config_composer
        self._tts_client = None

    def _get_tts_client(self):
        """TTS 클라이언트 가져오기 (지연 로딩)"""
        if self._tts_client is None:
            try:
                self._tts_client = TTSFactory.create_tts_client(self.config_composer)
            except (ImportError, ValueError, RuntimeError) as e:
                logger.error("Failed to create TTS client: %s", e)
                raise HTTPException(status_code=500, detail=f"TTS service initialization failed: {str(e)}") from e
        return self._tts_client

    async def generate_speech(self, request: TTSRequest) -> bytes:
        """
        텍스트를 음성으로 변환

        Args:
            request: TTS 요청 데이터

        Returns:
            생성된 오디오 데이터 (bytes)

        Raises:
            HTTPException: TTS 처리 실패 시
        """
        try:
            tts_client = self._get_tts_client()

            # TTS 서비스 사용 가능성 확인
            if not await tts_client.is_available():
                raise HTTPException(status_code=503, detail="TTS service is not available")

            # 감정 리스트 생성 (순서: Happiness, Sadness, Disgust, Fear, Surprise, Anger, Other, Neutral)
            emotion_list = [
                request.happiness,
                request.sadness,
                request.disgust,
                request.fear,
                request.surprise,
                request.anger,
                request.other,
                request.neutral
            ]

            # TTS 클라이언트 제공자 정보 확인
            provider_info = tts_client.get_provider_info()
            is_zonos = provider_info.get("provider", "").lower() == "zonos"

            # 음성 생성 (zonos인 경우에만 emotion 전달)
            if is_zonos:
                audio_data = await tts_client.generate_speech(
                    text=request.text,
                    speaker=request.speaker,
                    language="ko",
                    output_format=request.output_format,
                    emotion=emotion_list
                )
            else:
                audio_data = await tts_client.generate_speech(
                    text=request.text,
                    speaker=request.speaker,
                    language="ko",
                    output_format=request.output_format
                )

            logger.info("TTS generation successful for text: %s...", request.text[:50])
            return audio_data

        except HTTPException as e:
            raise e
        except (RuntimeError, ValueError, ImportError) as e:
            logger.error("TTS generation failed: %s", e)
            raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}") from e

    async def get_tts_info(self) -> dict:
        """
        TTS 서비스 정보 반환

        Returns:
            TTS 서비스 정보
        """
        try:
            tts_client = self._get_tts_client()
            info = tts_client.get_provider_info()
            info["is_available"] = await tts_client.is_available()
            return info

        except (ImportError, ValueError, RuntimeError) as e:
            logger.error("Failed to get TTS info: %s", e)
            return {
                "provider": "unknown",
                "model": "unknown",
                "available": False,
                "error": str(e)
            }

    async def get_available_providers(self) -> dict:
        """
        사용 가능한 TTS 제공자 목록 반환

        Returns:
            제공자 목록
        """
        return TTSFactory.get_available_providers()

    async def cleanup(self):
        """리소스 정리"""
        try:
            if self._tts_client:
                await self._tts_client.cleanup()
                self._tts_client = None
            await TTSFactory.cleanup_instance()
        except (RuntimeError, AttributeError) as e:
            logger.warning("Error during TTS controller cleanup: %s", e)

@router.post("/generate",
             summary="텍스트를 음성으로 변환",
             description="주어진 텍스트를 음성 파일로 변환합니다.")
async def generate_speech(
    tts_request: TTSRequest,
    request: Request
):
    """
    텍스트를 음성으로 변환하여 오디오 파일로 반환

    Args:
        tts_request: TTS 요청 데이터

    Returns:
        오디오 파일 (streaming response)
    """
    try:
        tts_controller = get_tts_service(request)
        audio_data = await tts_controller.generate_speech(
            text=tts_request.text,
            speaker=tts_request.speaker,
            output_format=tts_request.output_format,
            language="ko",
        )

        # MIME 타입 설정
        media_type_map = {
            "wav": "audio/wav",
            "mp3": "audio/mpeg",
            "ogg": "audio/ogg"
        }
        media_type = media_type_map.get(tts_request.output_format.lower(), "audio/wav")

        # 스트리밍 응답으로 반환
        return StreamingResponse(
            io.BytesIO(audio_data),
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename=generated_speech.{tts_request.output_format}"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Unexpected error in TTS generation: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error during TTS generation") from e

@router.get("/info",
           summary="TTS 서비스 정보 조회",
           description="현재 TTS 서비스의 설정과 상태 정보를 반환합니다.")
async def get_provider_info(request: Request):
    """
    TTS 서비스 정보 반환

    Returns:
        TTS 서비스 상태 및 설정 정보
    """
    try:
        tts_controller = get_tts_service(request)
        return tts_controller.get_provider_info()
    except Exception as e:
        logger.error("Error retrieving TTS info: %s", e)
        raise HTTPException(status_code=500, detail="Failed to retrieve TTS service information") from e

@router.get("/simple-status")
async def get_tts_simple_status(request: Request):
    """
    TTS 서비스 상태 확인

    Returns:
        TTS 서비스 상태 정보
    """
    try:
        config_composer = get_config_composer(request)
        tts_service = get_tts_service(request)
        provider_info = tts_service.get_provider_info()
        is_available = config_composer.get_config_by_name("IS_AVAILABLE_TTS").value

        return JSONResponse(content={
            "available": is_available,
            "provider": provider_info.get("provider"),
            "model": provider_info.get("model"),
            "api_key_configured": provider_info.get("api_key_configured", False)
        })

    except Exception as e:
        logger.error("Error getting TTS status: %s", e)
        return JSONResponse(content={
            "available": False,
            "provider": None,
            "model": None,
            "error": str(e)
        })

@router.post("/refresh")
async def refresh_tts_factory(request: Request):
    val_superuser = await validate_superuser(request)
    if val_superuser.get("superuser") is not True:
        raise HTTPException(
            status_code=403,
            detail="Admin privileges required"
        )
    try:
        config_composer = get_config_composer(request)
        if config_composer.get_config_by_name("IS_AVAILABLE_TTS").value:
            tts_client = TTSFactory.create_tts_client(config_composer)
            request.app.state.tts_service = tts_client
            return {
                "message": "TTS configuration refreshed successfully"
            }
        else:
            if hasattr(request.app.state, 'tts_service') and request.app.state.tts_service is not None:
                try:
                    await request.app.state.tts_service.cleanup()
                except Exception as cleanup_e:
                    logger.warning(f"Error during existing TTS service cleanup: {cleanup_e}")

                request.app.state.tts_service = None
                import gc
                gc.collect()
            else:
                request.app.state.tts_service = None
            return {
                "message": "TTS service is disabled in configuration"
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to refresh TTS config: {str(e)}")

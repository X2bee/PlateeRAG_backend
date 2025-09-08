"""
Admin 관련 라우터 통합
"""
from fastapi import APIRouter
from .sttController import router as sttRouter
from .ttsController import router as ttsRouter


# Audio 라우터 통합
audio_router = APIRouter(prefix="/api", tags=["Audio"])

audio_router.include_router(sttRouter)
audio_router.include_router(ttsRouter)

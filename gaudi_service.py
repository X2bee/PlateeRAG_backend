"""
Gaudi VLLM 전용 마이크로서비스
원본 gaudiController의 모든 기능을 독립 서비스로 제공
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
import os
import sys

# VLLM 경로를 Python path에 추가
sys.path.insert(0, '/workspace/vllm')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("gaudi-vllm-service")

app = FastAPI(
    title="Gaudi VLLM Service",
    description="Dedicated service for Habana Gaudi HPU VLLM management",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 원본 gaudiController import (늦은 import로 의존성 문제 해결)
try:
    from controller.gaudiController_original import router as gaudiRouter
    app.include_router(gaudiRouter)
    logger.info("Gaudi controller loaded successfully")
except ImportError as e:
    logger.error(f"Failed to import Gaudi controller: {e}")
    # 기본 라우터 생성
    from fastapi import APIRouter
    
    fallback_router = APIRouter(prefix="/api/gaudi", tags=["gaudi-fallback"])
    
    @fallback_router.get("/health")
    async def fallback_health():
        return {
            "status": "error",
            "message": "Gaudi controller not available",
            "error": str(e)
        }
    
    app.include_router(fallback_router)

@app.get("/health")
async def service_health():
    """서비스 헬스 체크"""
    return {
        "service": "gaudi-vllm",
        "status": "healthy",
        "version": "1.0.0",
        "vllm_path": "/workspace/vllm"
    }

@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "message": "Gaudi VLLM Service",
        "docs": "/docs",
        "health": "/health"
    }

if __name__ == "__main__":
    host = os.environ.get("GAUDI_HOST", "0.0.0.0")
    port = int(os.environ.get("GAUDI_PORT", "8080"))
    debug = os.environ.get("DEBUG_MODE", "false").lower() in ('true', '1', 'yes', 'on')
    
    logger.info(f"Starting Gaudi VLLM Service on {host}:{port}")
    logger.info(f"VLLM path: /workspace/vllm")
    logger.info(f"Python path: {sys.path[:3]}...")
    
    if debug:
        uvicorn.run("gaudi_service:app", host=host, port=port, reload=True)
    else:
        uvicorn.run("gaudi_service:app", host=host, port=port, reload=False)
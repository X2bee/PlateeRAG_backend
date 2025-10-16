"""
Tool 관련 라우터 통합
"""
from fastapi import APIRouter
from controller.tools import toolStorageController, toolStoreController

# Tool 라우터 통합
tool_router = APIRouter(prefix="/api/tools", tags=["Tools"])

# Tool Storage 엔드포인트 (사용자 툴 관리)
tool_router.include_router(
    toolStorageController.router,
    prefix="/storage",
    tags=["Tool Storage"]
)

# Tool Store 엔드포인트 (툴 스토어 관리)
tool_router.include_router(
    toolStoreController.router,
    prefix="/store",
    tags=["Tool Store"]
)

"""
Admin 관련 라우터 통합
"""
from fastapi import APIRouter
from .adminBaseController import router as adminBaseRouter
from .adminUserController import router as adminUserRouter

# Admin 라우터 통합
admin_router = APIRouter(prefix="/api/admin", tags=["Admin"])

# 각 admin 라우터들을 통합 라우터에 포함
admin_router.include_router(adminBaseRouter)
admin_router.include_router(adminUserRouter)

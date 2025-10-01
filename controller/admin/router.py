"""
Admin 관련 라우터 통합
"""
from fastapi import APIRouter
from .adminBaseController import router as adminBaseRouter
from .adminUserController import router as adminUserRouter
from .adminWorkflowController import router as adminWorkflowRouter
from .adminSystemController import router as adminSystemController
from .adminGroupController import router as adminGroupRouter
from .adminDBController import router as adminDBRouter
from .adminPromptController import router as adminPromptController

# Admin 라우터 통합
admin_router = APIRouter(prefix="/api/admin", tags=["Admin"])

# 각 admin 라우터들을 통합 라우터에 포함
admin_router.include_router(adminBaseRouter)
admin_router.include_router(adminUserRouter)
admin_router.include_router(adminWorkflowRouter)
admin_router.include_router(adminSystemController)
admin_router.include_router(adminGroupRouter)
admin_router.include_router(adminDBRouter)
admin_router.include_router(adminPromptController)

"""
Manager 관련 라우터 통합
"""
from fastapi import APIRouter
from .managerBaseController import router as managerBaseRouter
from .managerUserController import router as managerUserRouter
from .managerGroupController import router as managerGroupRouter
from .managerWorkflowController import router as managerWorkflowRouter

# Manager 라우터 통합
manager_router = APIRouter(prefix="/api/manager", tags=["Manager"])

# 각 manager 라우터들을 통합 라우터에 포함
manager_router.include_router(managerBaseRouter)
manager_router.include_router(managerUserRouter)
manager_router.include_router(managerGroupRouter)
manager_router.include_router(managerWorkflowRouter)

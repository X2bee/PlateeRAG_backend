"""
Admin 관련 라우터 통합
"""
from fastapi import APIRouter
from .workflowController import router as workflowRouter
from .workflowDeployController import router as workflowDeployRouter

# Admin 라우터 통합
workflow_router = APIRouter(prefix="/api/workflow", tags=["Workflow"])

# 각 admin 라우터들을 통합 라우터에 포함
workflow_router.include_router(workflowRouter)
workflow_router.include_router(workflowDeployRouter)

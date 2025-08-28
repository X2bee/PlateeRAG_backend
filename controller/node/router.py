"""
Admin 관련 라우터 통합
"""
from fastapi import APIRouter
from .nodeApiController import router as nodeApiRouter
from .nodeController import router as nodeRouter

# Node 라우터 통합
node_router = APIRouter(prefix="/api", tags=["Node"])

node_router.include_router(nodeApiRouter)
node_router.include_router(nodeRouter)

"""
Admin 관련 라우터 통합
"""
from fastapi import APIRouter
from .nodeApiController import router as nodeApiRouter
from .nodeController import router as nodeRouter
from .nodeStateController import router as nodeStateRouter

# Node 라우터 통합
node_router = APIRouter(prefix="/api", tags=["Node"])

node_router.include_router(nodeApiRouter)
node_router.include_router(nodeRouter)
node_router.include_router(nodeStateRouter)

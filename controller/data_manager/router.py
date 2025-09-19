from fastapi import APIRouter
from .dataManagerBaseController import router as dataManagerBase
from .dataManagerProcessingController import router as dataManagerProcessing
# Audio 라우터 통합
data_manager_router = APIRouter(prefix="/api/data-manager", tags=["Audio"])

data_manager_router.include_router(dataManagerBase)
data_manager_router.include_router(dataManagerProcessing)

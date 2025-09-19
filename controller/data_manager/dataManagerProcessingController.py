from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from controller.helper.singletonHelper import get_db_manager, get_data_manager_registry
from controller.helper.controllerHelper import extract_user_id_from_request

router = APIRouter(prefix="/processing", tags=["data-manager"])
logger = logging.getLogger("data-manager-controller")

# ========== Request Models ==========
class DownloadDatasetRequest(BaseModel):
    """데이터셋 다운로드 요청"""
    manager_id: str = Field(..., description="매니저 ID")
    repo_id: str = Field(..., description="Huggingface 리포지토리 ID", example="squad")
    filename: Optional[str] = Field(None, description="특정 파일명 (선택사항)")
    split: Optional[str] = Field(None, description="데이터 분할", example="train")

# ========== Helper Functions ==========
def get_manager_with_auth(registry, manager_id: str, user_id: str):
    """인증된 매니저 가져오기"""
    manager = registry.get_manager(manager_id, user_id)
    if not manager:
        raise HTTPException(
            status_code=404,
            detail=f"매니저 '{manager_id}'를 찾을 수 없거나 접근 권한이 없습니다"
        )
    return manager

# ========== API Endpoints ==========

@router.post("/hf/download-dataset",
    summary="데이터셋 다운로드 및 적재",
    description="Huggingface 리포지토리에서 데이터셋을 다운로드하고 pyarrow로 적재합니다.",
    response_model=Dict[str, Any])
async def download_dataset(request: Request, download_request: DownloadDatasetRequest) -> Dict[str, Any]:
    """데이터셋 다운로드 및 적재"""
    try:
        user_id = extract_user_id_from_request(request)
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID가 제공되지 않았습니다")

        registry = get_data_manager_registry(request)
        manager = get_manager_with_auth(registry, download_request.manager_id, user_id)

        # 데이터셋 다운로드 및 적재
        result = manager.download_and_load_dataset(
            repo_id=download_request.repo_id,
            filename=download_request.filename,
            split=download_request.split
        )

        logger.info(f"Dataset downloaded and loaded for manager {download_request.manager_id} from {download_request.repo_id}")

        return {
            "success": True,
            "manager_id": download_request.manager_id,
            "message": "데이터셋이 성공적으로 다운로드되고 적재되었습니다",
            "download_info": result
        }

    except HTTPException:
        raise
    except RuntimeError as e:
        logger.error(f"데이터셋 다운로드 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"예상치 못한 오류: {e}")
        raise HTTPException(status_code=500, detail="데이터셋 다운로드 중 오류가 발생했습니다")

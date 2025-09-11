"""
워크플로우 배포 관련 엔드포인트들
"""
import secrets
import string
import logging
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from controller.helper.controllerHelper import extract_user_id_from_request
from controller.helper.singletonHelper import get_db_manager
from controller.workflow.models.requests import DeployToggleRequest, DeployStatusRequest
from service.database.models.deploy import DeployMeta

logger = logging.getLogger("deploy-endpoints")
router = APIRouter()

@router.post("/status/{workflow_name}")
async def get_deploy_status(request: Request, workflow_name: str, staus_request: DeployStatusRequest):
    """
    특정 워크플로우의 배포 상태를 확인합니다.

    Args:
        workflow_name: 워크플로우 이름

    Returns:
        배포 상태 정보 (is_deployed, deploy_key 등)
    """
    try:
        if not staus_request.user_id:
            user_id = extract_user_id_from_request(request)
        else:
            user_id = staus_request.user_id
        app_db = get_db_manager(request)

        # DeployMeta 조회
        deploy_data = app_db.find_by_condition(
            DeployMeta,
            {
                "user_id": user_id,
                "workflow_name": workflow_name,
            },
            limit=1
        )

        if not deploy_data:
            raise HTTPException(status_code=404, detail="배포 메타데이터를 찾을 수 없습니다")

        deploy_meta = deploy_data[0]

        response_data = {
            "workflow_name": workflow_name,
            "workflow_id": deploy_meta.workflow_id,
            "is_deployed": deploy_meta.is_deployed,
            "deploy_key": deploy_meta.deploy_key if deploy_meta.is_deployed else None,
            "created_at": deploy_meta.created_at.isoformat() if hasattr(deploy_meta, 'created_at') and deploy_meta.created_at else None,
            "updated_at": deploy_meta.updated_at.isoformat() if hasattr(deploy_meta, 'updated_at') and deploy_meta.updated_at else None
        }

        return JSONResponse(content=response_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving deploy status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"배포 상태 조회 실패: {str(e)}")

# @router.post("/toggle/{workflow_name}")
# async def toggle_deploy_status(request: Request, workflow_name: str, toggle_request: DeployToggleRequest):
#     """
#     워크플로우의 배포 상태를 활성화/비활성화합니다.

#     Args:
#         workflow_name: 워크플로우 이름
#         toggle_request: 배포 토글 요청 (enable: 활성화 여부)

#     Returns:
#         업데이트된 배포 상태 정보
#     """
#     try:
#         user_id = extract_user_id_from_request(request)
#         app_db = get_db_manager(request)

#         # DeployMeta 조회
#         deploy_data = app_db.find_by_condition(
#             DeployMeta,
#             {
#                 "user_id": user_id,
#                 "workflow_name": workflow_name,
#             },
#             limit=1
#         )

#         if not deploy_data:
#             raise HTTPException(status_code=404, detail="배포 메타데이터를 찾을 수 없습니다")

#         deploy_meta = deploy_data[0]

#         # 배포 상태 업데이트
#         deploy_meta.is_deployed = toggle_request.enable

#         if toggle_request.enable:
#             # 배포 활성화 시 새로운 deploy_key 생성
#             # 32자리 랜덤 키 생성 (영문자 + 숫자)
#             alphabet = string.ascii_letters + string.digits
#             deploy_key = ''.join(secrets.choice(alphabet) for _ in range(32))
#             deploy_meta.deploy_key = deploy_key

#             logger.info(f"Generated new deploy key for workflow: {workflow_name}")
#         else:
#             # 배포 비활성화 시 deploy_key 초기화
#             deploy_meta.deploy_key = ""
#             logger.info(f"Cleared deploy key for workflow: {workflow_name}")

#         # DB 업데이트
#         update_result = app_db.update(deploy_meta)

#         if not update_result or update_result.get("result") != "success":
#             raise HTTPException(
#                 status_code=500,
#                 detail=f"배포 상태 업데이트 실패: {update_result.get('error', 'Unknown error')}"
#             )

#         response_data = {
#             "workflow_name": workflow_name,
#             "workflow_id": deploy_meta.workflow_id,
#             "is_deployed": deploy_meta.is_deployed,
#             "deploy_key": deploy_meta.deploy_key if deploy_meta.is_deployed else None,
#             "message": f"배포가 {'활성화' if toggle_request.enable else '비활성화'}되었습니다",
#             "updated_at": deploy_meta.updated_at.isoformat() if hasattr(deploy_meta, 'updated_at') and deploy_meta.updated_at else None
#         }

#         logger.info(f"Deploy status {'enabled' if toggle_request.enable else 'disabled'} for workflow: {workflow_name}")
#         return JSONResponse(content=response_data)

#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Error toggling deploy status: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"배포 상태 변경 실패: {str(e)}")

@router.get("/key/{workflow_name}")
async def get_deploy_key(request: Request, workflow_name: str):
    """
    특정 워크플로우의 배포 키값을 조회합니다.

    Args:
        workflow_name: 워크플로우 이름

    Returns:
        배포 키값 (배포가 활성화된 경우에만)
    """
    try:
        user_id = extract_user_id_from_request(request)
        app_db = get_db_manager(request)

        # DeployMeta 조회
        deploy_data = app_db.find_by_condition(
            DeployMeta,
            {
                "user_id": user_id,
                "workflow_name": workflow_name,
            },
            limit=1
        )

        if not deploy_data:
            raise HTTPException(status_code=404, detail="배포 메타데이터를 찾을 수 없습니다")

        deploy_meta = deploy_data[0]

        # 배포가 비활성화된 경우 키값을 제공하지 않음
        if not deploy_meta.is_deployed:
            raise HTTPException(status_code=403, detail="배포가 비활성화된 워크플로우입니다")

        # 배포 키가 없는 경우 (예외적인 상황)
        if not deploy_meta.deploy_key:
            raise HTTPException(status_code=500, detail="배포 키가 생성되지 않았습니다")

        response_data = {
            "workflow_name": workflow_name,
            "workflow_id": deploy_meta.workflow_id,
            "deploy_key": deploy_meta.deploy_key,
            "is_deployed": deploy_meta.is_deployed,
            "message": "배포 키 조회 성공"
        }

        logger.info(f"Deploy key retrieved for workflow: {workflow_name}")
        return JSONResponse(content=response_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving deploy key: {str(e)}")
        raise HTTPException(status_code=500, detail=f"배포 키 조회 실패: {str(e)}")

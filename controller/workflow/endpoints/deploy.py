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
from service.database.logger_helper import create_logger

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
    if not staus_request.user_id:
        user_id = extract_user_id_from_request(request)
    else:
        user_id = staus_request.user_id

    if not user_id:
        raise HTTPException(status_code=400, detail="User ID not found in request")

    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, user_id, request)

    try:
        backend_log.info("Starting deploy status check",
                        metadata={"workflow_name": workflow_name,
                                "requested_user_id": staus_request.user_id})

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
            backend_log.warn("Deploy metadata not found",
                           metadata={"workflow_name": workflow_name})
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

        backend_log.success("Deploy status retrieved successfully",
                          metadata={"workflow_name": workflow_name,
                                  "workflow_id": deploy_meta.workflow_id,
                                  "is_deployed": deploy_meta.is_deployed,
                                  "has_deploy_key": bool(deploy_meta.deploy_key),
                                  "is_accepted": getattr(deploy_meta, 'is_accepted', None),
                                  "inquire_deploy": getattr(deploy_meta, 'inquire_deploy', None)})

        return JSONResponse(content=response_data)

    except HTTPException:
        raise
    except Exception as e:
        backend_log.error("Failed to retrieve deploy status", exception=e,
                         metadata={"workflow_name": workflow_name})
        logger.error(f"Error retrieving deploy status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"배포 상태 조회 실패: {str(e)}")


@router.get("/key/{workflow_name}")
async def get_deploy_key(request: Request, workflow_name: str):
    """
    특정 워크플로우의 배포 키값을 조회합니다.

    Args:
        workflow_name: 워크플로우 이름

    Returns:
        배포 키값 (배포가 활성화된 경우에만)
    """
    user_id = extract_user_id_from_request(request)
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID not found in request")

    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, user_id, request)

    try:
        backend_log.info("Starting deploy key retrieval",
                        metadata={"workflow_name": workflow_name})

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
            backend_log.warn("Deploy metadata not found for key retrieval",
                           metadata={"workflow_name": workflow_name})
            raise HTTPException(status_code=404, detail="배포 메타데이터를 찾을 수 없습니다")

        deploy_meta = deploy_data[0]

        # 배포가 비활성화된 경우 키값을 제공하지 않음
        if not deploy_meta.is_deployed:
            backend_log.warn("Deploy key requested for inactive deployment",
                           metadata={"workflow_name": workflow_name, "is_deployed": False})
            raise HTTPException(status_code=403, detail="배포가 비활성화된 워크플로우입니다")

        # 배포 키가 없는 경우 (예외적인 상황)
        if not deploy_meta.deploy_key:
            backend_log.error("Deploy key not generated for active deployment",
                            metadata={"workflow_name": workflow_name, "is_deployed": True})
            raise HTTPException(status_code=500, detail="배포 키가 생성되지 않았습니다")

        response_data = {
            "workflow_name": workflow_name,
            "workflow_id": deploy_meta.workflow_id,
            "deploy_key": deploy_meta.deploy_key,
            "is_deployed": deploy_meta.is_deployed,
            "message": "배포 키 조회 성공"
        }

        backend_log.success("Deploy key retrieved successfully",
                          metadata={"workflow_name": workflow_name,
                                  "workflow_id": deploy_meta.workflow_id,
                                  "is_deployed": deploy_meta.is_deployed,
                                  "deploy_key_length": len(deploy_meta.deploy_key)})

        logger.info(f"Deploy key retrieved for workflow: {workflow_name}")
        return JSONResponse(content=response_data)

    except HTTPException:
        raise
    except Exception as e:
        backend_log.error("Failed to retrieve deploy key", exception=e,
                         metadata={"workflow_name": workflow_name})
        logger.error(f"Error retrieving deploy key: {str(e)}")
        raise HTTPException(status_code=500, detail=f"배포 키 조회 실패: {str(e)}")

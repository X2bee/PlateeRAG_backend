"""
기본 워크플로우 CRUD 작업 관련 엔드포인트들
"""
import os
import json
import copy
from datetime import datetime
import logging
import secrets
import string
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from controller.helper.controllerHelper import extract_user_id_from_request, require_admin_access
from controller.helper.singletonHelper import get_db_manager, get_config_composer
from controller.workflow.models.requests import SaveWorkflowRequest
from service.database.models.executor import ExecutionMeta, ExecutionIO
from controller.helper.utils.auth_helpers import workflow_user_id_extractor
from controller.helper.utils.data_parsers import parse_input_data
from service.database.models.user import User
from service.database.models.workflow import WorkflowMeta, WorkflowStoreMeta, WorkflowVersion
from service.database.models.deploy import DeployMeta
from service.database.logger_helper import create_logger

logger = logging.getLogger("workflow-store-endpoints")
router = APIRouter()

@router.get("/list")
async def list_workflows(request: Request):
    user_id = extract_user_id_from_request(request)
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID not found in request")

    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, user_id, request)

    try:
        workflow_data = app_db.find_all(WorkflowStoreMeta, limit= 100000, join_user=True, ignore_columns=['workflow_data'])
        backend_log.success("Workflow list retrieved successfully",
                          metadata={"workflow_count": len(workflow_data),
                                  "workflow_files": [wf.workflow_name for wf in workflow_data[:10]]})  # 처음 10개만 로깅

        logger.info(f"Found {len(workflow_data)} workflow files")
        return {"workflows": workflow_data}

    except Exception as e:
        backend_log.error("Failed to list workflows", exception=e)
        logger.error(f"Error listing workflows: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list workflows: {str(e)}")

@router.get("/duplicate/{workflow_name}")
async def duplicate_workflow(request: Request, workflow_name: str, workflow_upload_name: str = None, user_id: str = None, current_version: float = None):
    """
    특정 workflow를 복제합니다.
    """
    try:
        login_user_id = extract_user_id_from_request(request)
        if not login_user_id:
            raise HTTPException(status_code=400, detail="User ID not found in request")
        app_db = get_db_manager(request)
        backend_log = create_logger(app_db, login_user_id, request)

        store_workflow_data = app_db.find_by_condition(
            WorkflowStoreMeta,
            {
                "user_id": user_id,
                "workflow_name": workflow_name,
                "workflow_upload_name": workflow_upload_name,
                "current_version": current_version,
            },
            limit=1
        )
        if not store_workflow_data:
            raise HTTPException(status_code=404, detail="Workflow not found")
        store_workflow_data = store_workflow_data[0]

        copy_workflow_name = f"{store_workflow_data.workflow_upload_name}_copy"
        existing_copy_data = app_db.find_by_condition(WorkflowMeta, {"user_id": login_user_id, "workflow_name": copy_workflow_name}, limit=1)
        if existing_copy_data:
            logger.warning(f"Workflow already exists for user '{login_user_id}': {copy_workflow_name}. Change target file name.")
            counter = 1
            while existing_copy_data:
                copy_workflow_name = f"{workflow_name}_copy_{counter}"
                existing_copy_data = app_db.find_by_condition(WorkflowMeta, {"user_id": login_user_id, "workflow_name": copy_workflow_name}, limit=1)
                counter += 1

        version_new = round(1.0, 3)
        workflow_meta = WorkflowMeta(
            user_id=login_user_id,
            workflow_id=store_workflow_data.workflow_id,
            workflow_name=copy_workflow_name,
            node_count=store_workflow_data.node_count,
            edge_count=store_workflow_data.edge_count,
            has_startnode=store_workflow_data.has_startnode,
            has_endnode=store_workflow_data.has_endnode,
            is_completed=(store_workflow_data.has_startnode and store_workflow_data.has_endnode),
            workflow_data=store_workflow_data.workflow_data,
            current_version=version_new,
            latest_version=version_new,
        )

        insert_result = app_db.insert(workflow_meta)
        if insert_result and insert_result.get("result") == "success":
            deploy_meta = DeployMeta(
                user_id=login_user_id,
                workflow_id=store_workflow_data.workflow_id,
                workflow_name=copy_workflow_name,
                is_deployed=False,
                is_accepted=True,
                inquire_deploy=False,
                deploy_key=""
            )
            app_db.insert(deploy_meta)

            workflow_version_data = WorkflowVersion(
                user_id=login_user_id,
                workflow_meta_id=workflow_meta.id,
                workflow_id=workflow_meta.workflow_id,
                workflow_name=workflow_meta.workflow_name,
                version=version_new,
                current_use=True,
                workflow_data=store_workflow_data.workflow_data,
                version_label=f"v{version_new}"
            )
            app_db.insert(workflow_version_data)
        backend_log.info("Workflow duplicated successfully",
                        metadata={"workflow_name": workflow_name,
                                  "workflow_upload_name": workflow_upload_name,
                                  "new_version": version_new,
                        })

        logger.info(f"Workflow duplicated successfully: {workflow_upload_name}")
        return {"success": True, "message": f"Workflow '{workflow_upload_name}' duplicated successfully"}

    except Exception as e:
        logger.error(f"Error loading workflow: {str(e)}")
        backend_log.error("Failed to duplicate workflow", exception=e,
                         metadata={"workflow_name": workflow_name,
                                   "workflow_upload_name": workflow_upload_name if 'workflow_upload_name' in locals() else None,
                                   "user_id": user_id if 'user_id' in locals() else None,
                                   "current_version": current_version if 'current_version' in locals() else None,
                        })
        raise HTTPException(status_code=500, detail=f"Failed to load workflow: {str(e)}")

@router.post("/update/{workflow_name}")
async def update_workflow(request: Request, workflow_name: str, update_dict: dict):
    user_id = extract_user_id_from_request(request)
    app_db = get_db_manager(request)
    config_composer = get_config_composer(request)

    try:
        existing_data = app_db.find_by_condition(
            WorkflowMeta,
            {
                "user_id": user_id,
                "workflow_name": workflow_name
            },
            limit=1
        )

        if not existing_data:
            raise HTTPException(status_code=404, detail="Workflow not found")
        existing_data = existing_data[0]

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
        if not deploy_data[0].is_accepted:
            raise HTTPException(status_code=400, detail="해당 워크플로우에 대한 권한이 박탈되었습니다. 편집할 수 없습니다.")

        deploy_meta = deploy_data[0]

        existing_data.is_shared = update_dict.get("is_shared", existing_data.is_shared)
        existing_data.share_group = update_dict.get("share_group", existing_data.share_group)
        existing_data.share_permissions = update_dict.get("share_permissions", existing_data.share_permissions)

        deploy_enabled = update_dict.get("enable_deploy", deploy_meta.is_deployed)
        deploy_meta.is_deployed = deploy_enabled
        free_deploy_mode = config_composer.get_config_by_name("FREE_CHAT_DEPLOYMENT_MODE").value

        if deploy_enabled:
            if not deploy_meta.deploy_key or deploy_meta.deploy_key.strip() == "":
                alphabet = string.ascii_letters + string.digits
                deploy_key = ''.join(secrets.choice(alphabet) for _ in range(32))
                deploy_meta.deploy_key = deploy_key

            if not free_deploy_mode:
                try:
                    admin_access = require_admin_access(request)
                except:
                    deploy_meta.is_deployed = False
                    deploy_meta.inquire_deploy = True

            logger.info(f"Generated new deploy key for workflow: {workflow_name}")
        else:
            deploy_meta.deploy_key = ""
            deploy_meta.inquire_deploy = False
            logger.info(f"Cleared deploy key for workflow: {workflow_name}")

        app_db.update(existing_data)
        app_db.update(deploy_meta)

        return {
            "message": "Workflow updated successfully",
            "workflow_name": existing_data.workflow_name,
            "deploy_key": deploy_meta.deploy_key if deploy_meta.is_deployed else None,
            "inquire_deploy": deploy_meta.inquire_deploy,
        }

    except Exception as e:
        logger.error(f"Failed to update workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update workflow: {str(e)}")

@router.delete("/delete/{workflow_name}")
async def delete_workflow(request: Request, workflow_name: str, workflow_upload_name: str, current_version: float):
    """
    특정 workflow를 삭제합니다.
    """
    user_id = extract_user_id_from_request(request)
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID not found in request")

    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, user_id, request)

    try:
        backend_log.info("Starting workflow deletion",
                        metadata={"workflow_name": workflow_name})

        existing_data = app_db.find_by_condition(
            WorkflowStoreMeta,
            {
                "user_id": user_id,
                "workflow_name": workflow_name,
                "workflow_upload_name": workflow_upload_name,
                "current_version": current_version,
            },
            ignore_columns=['workflow_data'],
            limit=1
        )

        if not existing_data:
            backend_log.warn("Workflow metadata not found for deletion",
                           metadata={"workflow_name": workflow_name})
            raise HTTPException(status_code=404, detail=f"Workflow '{workflow_name}' not found")

        app_db.delete_by_condition(WorkflowStoreMeta, {
            "user_id": user_id,
            "workflow_name": workflow_name,
            "workflow_upload_name": workflow_upload_name,
            "current_version": current_version,
        })

        backend_log.success("Workflow deleted successfully",
                          metadata={"workflow_name": workflow_name,
                                  "workflow_upload_name": workflow_upload_name,
                                  "current_version": current_version})

        logger.info(f"Workflow deleted successfully: {workflow_name}")
        return JSONResponse(content={
            "success": True,
            "message": f"Workflow '{workflow_name}' deleted successfully"
        })

    except Exception as e:
        backend_log.error("Workflow deletion failed", exception=e,
                         metadata={"workflow_name": workflow_name})
        logger.error(f"Error deleting workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete workflow: {str(e)}")

@router.get("/list/detail")
async def list_workflows_detail(request: Request):
    """
    downloads 폴더에 있는 모든 workflow 파일들의 상세 정보를 반환합니다.
    각 워크플로우에 대해 파일명, workflow_id, 노드 수, 마지막 수정일자를 포함합니다.
    """
    user_id = extract_user_id_from_request(request)
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID not found in request")

    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, user_id, request)

    try:
        user = app_db.find_by_id(User, user_id)
        groups = user.groups
        user_name = user.username if user else "Unknown User"

        backend_log.info("Starting detailed workflow list retrieval",
                        metadata={"user_name": user_name, "groups": groups, "groups_count": len(groups) if groups else 0})

        # 직접 SQL 쿼리로 워크플로우와 사용자 정보 조인
        # 자신의 워크플로우
        own_workflows_query = """
            SELECT
                wm.id, wm.created_at, wm.updated_at,
                wm.user_id, wm.workflow_id, wm.workflow_name,
                wm.node_count, wm.edge_count, wm.has_startnode, wm.has_endnode,
                wm.is_completed, wm.metadata, wm.is_shared, wm.share_group, wm.share_permissions,
                u.username, u.full_name,
                dm.is_deployed, dm.deploy_key, dm.is_accepted, dm.inquire_deploy
            FROM workflow_meta wm
            LEFT JOIN users u ON wm.user_id = u.id
            LEFT JOIN deploy_meta dm ON wm.workflow_id = dm.workflow_id AND wm.workflow_name = dm.workflow_name AND wm.user_id = dm.user_id
            WHERE wm.user_id = %s
            ORDER BY wm.updated_at DESC
            LIMIT 10000
        """
        existing_data = app_db.config_db_manager.execute_query(own_workflows_query, (user_id,))
        own_workflow_count = len(existing_data) if existing_data else 0

        # 그룹 공유 워크플로우 추가
        shared_workflow_count = 0
        if groups and groups != None and groups != [] and len(groups) > 0:
            for group_name in groups:
                shared_workflows_query = """
                    SELECT
                        wm.id, wm.created_at, wm.updated_at,
                        wm.user_id, wm.workflow_id, wm.workflow_name,
                        wm.node_count, wm.edge_count, wm.has_startnode, wm.has_endnode,
                        wm.is_completed, wm.metadata, wm.is_shared, wm.share_group, wm.share_permissions,
                        u.username, u.full_name,
                        dm.is_deployed, dm.deploy_key, dm.is_accepted, dm.inquire_deploy
                    FROM workflow_meta wm
                    LEFT JOIN users u ON wm.user_id = u.id
                    LEFT JOIN deploy_meta dm ON wm.workflow_id = dm.workflow_id AND wm.workflow_name = dm.workflow_name AND wm.user_id = dm.user_id
                    WHERE wm.share_group = %s AND wm.is_shared = true
                    ORDER BY wm.updated_at DESC
                    LIMIT 10000
                """
                shared_data = app_db.config_db_manager.execute_query(shared_workflows_query, (group_name,))
                if shared_data:
                    existing_data.extend(shared_data)
                    shared_workflow_count += len(shared_data)

        seen_ids = set()
        unique_data = []
        for item in existing_data:
            if item.get("id") not in seen_ids:
                seen_ids.add(item.get("id"))
                item_dict = dict(item)
                if 'created_at' in item_dict and item_dict['created_at']:
                    item_dict['created_at'] = item_dict['created_at'].isoformat() if hasattr(item_dict['created_at'], 'isoformat') else str(item_dict['created_at'])
                if 'updated_at' in item_dict and item_dict['updated_at']:
                    item_dict['updated_at'] = item_dict['updated_at'].isoformat() if hasattr(item_dict['updated_at'], 'isoformat') else str(item_dict['updated_at'])
                unique_data.append(item_dict)

        backend_log.success("Detailed workflow list retrieved successfully",
                          metadata={"own_workflows": own_workflow_count,
                                  "shared_workflows": shared_workflow_count,
                                  "total_unique_workflows": len(unique_data),
                                  "user_name": user_name,
                                  "groups_count": len(groups) if groups else 0})

        logger.info(f"Found {len(unique_data)} workflow files with detailed information")

        return JSONResponse(content={"workflows": unique_data})

    except Exception as e:
        backend_log.error("Failed to retrieve detailed workflow list", exception=e,
                         metadata={"user_name": user_name if 'user_name' in locals() else None})
        logger.error(f"Error listing workflow details: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list workflow details: {str(e)}")

@router.post("/upload/{workflow_name}")
async def upload_workflow(request: Request, workflow_name: str, workflow_upload_name: str, description: str = "", tags: list = []):
    user_id = extract_user_id_from_request(request)
    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, user_id, request)

    try:
        existing_data = app_db.find_by_condition(
            WorkflowMeta,
            {
                "user_id": user_id,
                "workflow_name": workflow_name
            },
            limit=1
        )

        if not existing_data:
            raise HTTPException(status_code=404, detail="Workflow not found")
        existing_data = existing_data[0]

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
        if not deploy_data[0].is_accepted:
            raise HTTPException(status_code=400, detail="해당 워크플로우에 대한 권한이 박탈되었습니다. 편집할 수 없습니다.")

        if not workflow_upload_name or workflow_upload_name.strip() == "":
            raise HTTPException(status_code=400, detail="업로드할 워크플로우 이름을 입력하세요.")

        existing_upload_data = app_db.find_by_condition(
            WorkflowStoreMeta,
            {
                "user_id": user_id,
                "workflow_upload_name": workflow_upload_name,
                "current_version": existing_data.current_version,
            },
            limit=1
        )
        if existing_upload_data:
            raise HTTPException(status_code=400, detail="이미 동일한 이름의 업로드된 워크플로우가 존재합니다. 다른 이름을 사용하세요.")

        workflow_store_data = WorkflowStoreMeta(
            user_id=user_id,
            workflow_id=existing_data.workflow_id,
            workflow_name=existing_data.workflow_name,
            workflow_upload_name=workflow_upload_name,
            node_count=existing_data.node_count,
            edge_count=existing_data.edge_count,
            has_startnode=existing_data.has_startnode,
            has_endnode=existing_data.has_endnode,
            is_completed=existing_data.is_completed,
            metadata=existing_data.metadata,
            current_version=existing_data.current_version,
            latest_version=existing_data.latest_version,
            is_template=False,
            description=description,
            tags=tags,
            workflow_data=existing_data.workflow_data,
        )

        insert_result = app_db.insert(workflow_store_data)
        backend_log.info("Successfully uploaded workflow",
                        metadata={"workflow_name": workflow_name,
                                  "workflow_upload_name": workflow_upload_name,
                                  "description": description,
                        })

        return {
            "message": "Workflow uploaded successfully",
            "workflow_upload_name": workflow_upload_name,
        }

    except Exception as e:
        logger.error(f"Failed to upload workflow: {str(e)}")
        backend_log.error("Failed to upload workflow", exception=e,
                         metadata={"workflow_name": workflow_name,
                                   "workflow_upload_name": workflow_upload_name if 'workflow_upload_name' in locals() else None,
                                   "description": description if 'description' in locals() else None,
                        })
        raise HTTPException(status_code=500, detail=f"Failed to upload workflow: {str(e)}")

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
from typing import Any, Dict, Optional
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


def parse_json_field(value: Any, field_name: str = "field", workflow_id: Optional[int] = None) -> Any:
    """
    JSON 문자열 필드를 안전하게 파싱합니다.
    한글 유니코드 이스케이프 시퀀스도 올바르게 처리합니다.

    Args:
        value: 파싱할 값 (문자열 또는 이미 파싱된 객체)
        field_name: 필드 이름 (로깅용)
        workflow_id: 워크플로우 ID (로깅용)

    Returns:
        파싱된 Python 객체 또는 원본 값
    """
    if not value:
        return None

    # 이미 dict나 list인 경우 그대로 반환
    if isinstance(value, (dict, list)):
        return value

    # 문자열인 경우 JSON 파싱 시도
    if isinstance(value, str):
        try:
            # json.loads는 자동으로 유니코드 이스케이프를 처리함
            parsed = json.loads(value)
            return parsed
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(
                f"Failed to parse {field_name} for workflow {workflow_id}: {str(e)[:100]}"
            )
            return None

    # 그 외의 타입은 원본 반환
    return value


def normalize_workflow_dict(workflow_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    워크플로우 딕셔너리의 JSON 필드들을 정규화합니다.

    Args:
        workflow_dict: 워크플로우 데이터 딕셔너리

    Returns:
        정규화된 딕셔너리
    """
    workflow_id = workflow_dict.get('id')

    # workflow_data 파싱
    if 'workflow_data' in workflow_dict:
        workflow_dict['workflow_data'] = parse_json_field(
            workflow_dict['workflow_data'],
            'workflow_data',
            workflow_id
        )

    # metadata 파싱
    if 'metadata' in workflow_dict:
        workflow_dict['metadata'] = parse_json_field(
            workflow_dict['metadata'],
            'metadata',
            workflow_id
        )

    # tags 파싱
    if 'tags' in workflow_dict:
        workflow_dict['tags'] = parse_json_field(
            workflow_dict['tags'],
            'tags',
            workflow_id
        )

    # 날짜 필드 ISO 형식으로 변환
    for date_field in ['created_at', 'updated_at']:
        if date_field in workflow_dict and workflow_dict[date_field]:
            if hasattr(workflow_dict[date_field], 'isoformat'):
                workflow_dict[date_field] = workflow_dict[date_field].isoformat()
            elif not isinstance(workflow_dict[date_field], str):
                workflow_dict[date_field] = str(workflow_dict[date_field])

    return workflow_dict

@router.get("/list")
async def list_workflows(request: Request):
    user_id = extract_user_id_from_request(request)
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID not found in request")

    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, user_id, request)

    try:
        workflow_data = app_db.find_all(WorkflowStoreMeta, limit= 100000, join_user=True)

        # workflow_data를 파싱하여 JSON으로 변환
        parsed_workflows = []
        for wf in workflow_data:
            wf_dict = wf if isinstance(wf, dict) else wf.__dict__
            normalized_wf = normalize_workflow_dict(wf_dict)
            parsed_workflows.append(normalized_wf)

        backend_log.success("Workflow list retrieved successfully",
                          metadata={"workflow_count": len(parsed_workflows),
                                  "workflow_files": [wf.get('workflow_name') for wf in parsed_workflows[:10]]})

        logger.info(f"Found {len(parsed_workflows)} workflow files")
        return {"workflows": parsed_workflows}

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

        # 입력 파라미터 정규화
        if workflow_name:
            workflow_name = workflow_name.strip()
        if workflow_upload_name:
            workflow_upload_name = workflow_upload_name.strip()

        search_condition = {
            "workflow_name": workflow_name,
            "workflow_upload_name": workflow_upload_name,
            "current_version": current_version,
        }
        if user_id and user_id != "" and user_id != None:
            search_condition["user_id"] = user_id

        store_workflow_data = app_db.find_by_condition(
            WorkflowStoreMeta,
            search_condition,
            limit=1
        )
        if not store_workflow_data:
            raise HTTPException(status_code=404, detail="Workflow not found")
        store_workflow_data = store_workflow_data[0]

        # 한글이 포함된 워크플로우명 안전하게 처리
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

        # 한글 및 특수문자 정규화 처리
        workflow_upload_name = workflow_upload_name.strip()
        if description:
            description = description.strip()

        # tags 리스트의 각 항목 정규화
        if tags and isinstance(tags, list):
            tags = [tag.strip() if isinstance(tag, str) else str(tag) for tag in tags if tag]

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
            description=description if description else "",
            tags=tags if tags else [],
            workflow_data=existing_data.workflow_data,
        )

        insert_result = app_db.insert(workflow_store_data)
        backend_log.info("Successfully uploaded workflow",
                        metadata={"workflow_name": workflow_name,
                                  "workflow_upload_name": workflow_upload_name,
                                  "description": description,
                                  "tags_count": len(tags) if tags else 0,
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

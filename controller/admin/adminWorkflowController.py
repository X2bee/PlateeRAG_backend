import logging
import json
import string
import secrets
import os
from datetime import datetime
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse
from controller.helper.singletonHelper import get_db_manager
from controller.admin.adminBaseController import validate_superuser
from controller.workflow.utils.data_parsers import parse_input_data

from service.database.models.executor import ExecutionIO
from service.database.models.workflow import WorkflowMeta
from service.database.models.deploy import DeployMeta

logger = logging.getLogger("admin-workflow-controller")
router = APIRouter(prefix="/workflow", tags=["Admin"])

def extract_result_from_json(json_string):
    """
    Extract the 'result' field from a JSON string.
    If parsing fails or 'result' is not found, return the original string.
    """
    if not json_string:
        return json_string

    try:
        data = json.loads(json_string)
        return data.get("result", json_string)
    except (json.JSONDecodeError, TypeError):
        return json_string

def process_io_logs_efficient(io_logs):
    """
    Efficiently process io_logs using map and dictionary unpacking.
    """
    def process_single_log(log):
        log_dict = {k: v for k, v in log.__dict__.items() if not k.startswith('_')}
        log_dict.update({
            "input_data": extract_result_from_json(log.input_data),
            "output_data": extract_result_from_json(log.output_data)
        })
        return log_dict

    return list(map(process_single_log, io_logs))

@router.get("/admin-io-logs")
async def get_io_logs_by_id(request: Request, user_id = None, workflow_name: str = None, workflow_id: str = None):
    """
    관리자용 ExecutionIO 로그를 반환합니다. user_id가 없으면 모든 로그를 가져올 수 있습니다.
    """
    val_superuser = await validate_superuser(request)
    if val_superuser.get("superuser") is not True:
        raise HTTPException(
            status_code=403,
            detail="Admin privileges required"
        )

    try:
        app_db = get_db_manager(request)
        conditions = {}
        if user_id:
            conditions['user_id'] = user_id
        if workflow_name:
            conditions['workflow_name'] = workflow_name
        if workflow_id:
            conditions['workflow_id'] = workflow_id

        # 조건이 있으면 조건 검색, 없으면 모든 로그
        if conditions:
            result = app_db.find_by_condition(
                ExecutionIO,
                conditions,
                limit=1000000,
                orderby="updated_at",
                orderby_asc=True,
                return_list=True
            )
        else:
            raise HTTPException(
                status_code=400,
                detail="At least one filter parameter (user_id, workflow_name, workflow_id) must be provided"
            )

        if not result:
            logger.info("No IO logs found")
            return JSONResponse(content={
                "io_logs": [],
                "message": "No IO logs found"
            })

        io_logs = []
        for idx, row in enumerate(result):
            # input_data 파싱
            raw_input_data = json.loads(row['input_data']).get('result', None) if row['input_data'] else None
            parsed_input_data = parse_input_data(raw_input_data) if raw_input_data else None

            log_entry = {
                "log_id": idx + 1,
                "io_id": row['id'],
                "user_id": row['user_id'],
                "interaction_id": row['interaction_id'],
                "workflow_name": row['workflow_name'],
                "workflow_id": row['workflow_id'],
                "input_data": parsed_input_data,
                "output_data": json.loads(row['output_data']).get('result', None) if row['output_data'] else None,
                "updated_at": row['updated_at'].isoformat() if isinstance(row['updated_at'], datetime) else row['updated_at'],
                "created_at": row['created_at'].isoformat() if isinstance(row['created_at'], datetime) else row['created_at']
            }
            io_logs.append(log_entry)

        response_data = {
            "workflow_name": workflow_name,
            "workflow_id": workflow_id,
            "in_out_logs": io_logs,
            "message": "In/Out logs retrieved successfully"
        }

        return JSONResponse(content=response_data)

    except Exception as e:
        logger.error(f"Error fetching IO logs: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        ) from e

@router.get("/all-io-logs")
async def get_all_workflows_by_id(request: Request, page: int = 1, page_size: int = 250, user_id = None):
    val_superuser = await validate_superuser(request)
    if val_superuser.get("superuser") is not True:
        raise HTTPException(
            status_code=403,
            detail="Admin privileges required"
        )

    try:
        # 페이지 번호 검증
        if page < 1:
            page = 1
        if page_size < 1 or page_size > 1000:
            page_size = 250

        offset = (page - 1) * page_size

        app_db = get_db_manager(request)
        if user_id:
            io_logs = app_db.find_by_condition(ExecutionIO, {'user_id': user_id}, limit=page_size, offset=offset)
        else:
            io_logs = app_db.find_all(ExecutionIO, limit=page_size, offset=offset)
        processed_logs = process_io_logs_efficient(io_logs)

        return {
            "io_logs": processed_logs,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "offset": offset,
                "total_returned": len(processed_logs)
            }
        }
    except Exception as e:
        logger.error("Error fetching all IO logs: %s", str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        ) from e

@router.get("/all-list")
async def get_all_workflows(request: Request, page: int = 1, page_size: int = 250, user_id = None):
    val_superuser = await validate_superuser(request)
    if val_superuser.get("superuser") is not True:
        raise HTTPException(
            status_code=403,
            detail="Admin privileges required"
        )

    try:
        # 페이지 번호 검증
        if page < 1:
            page = 1
        if page_size < 1 or page_size > 1000:
            page_size = 250

        offset = (page - 1) * page_size

        app_db = get_db_manager(request)

        # 직접 SQL 쿼리 작성: workflow_meta와 users를 조인
        if user_id:
            query = """
                SELECT
                    wm.id, wm.created_at, wm.updated_at,
                    wm.user_id, wm.workflow_id, wm.workflow_name,
                    wm.node_count, wm.edge_count, wm.has_startnode, wm.has_endnode,
                    wm.is_completed, wm.metadata, wm.is_shared, wm.share_group, wm.share_permissions,
                    u.full_name
                FROM workflow_meta wm
                LEFT JOIN users u ON wm.user_id = u.id
                WHERE wm.user_id = %s
                ORDER BY wm.created_at DESC
                LIMIT %s OFFSET %s
            """
            all_workflows = app_db.config_db_manager.execute_query(query, (user_id, page_size, offset))
        else:
            query = """
                SELECT
                    wm.id, wm.created_at, wm.updated_at,
                    wm.user_id, wm.workflow_id, wm.workflow_name,
                    wm.node_count, wm.edge_count, wm.has_startnode, wm.has_endnode,
                    wm.is_completed, wm.metadata, wm.is_shared, wm.share_group, wm.share_permissions,
                    u.full_name, u.username
                FROM workflow_meta wm
                LEFT JOIN users u ON wm.user_id = u.id
                ORDER BY wm.created_at DESC
                LIMIT %s OFFSET %s
            """
            all_workflows = app_db.config_db_manager.execute_query(query, (page_size, offset))

        return {
            "workflows": all_workflows,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "offset": offset,
                "total_returned": len(all_workflows)
            }
        }
    except Exception as e:
        logger.error("Error fetching all IO logs: %s", str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        ) from e


@router.post("/update/{workflow_name}")
async def update_workflow(request: Request, workflow_name: str, update_dict: dict):
    validate_superuser(request)
    app_db = get_db_manager(request)
    user_id = update_dict.get("user_id")
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required in the update data")

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
        deploy_meta = deploy_data[0]

        existing_data.is_shared = update_dict.get("is_shared", existing_data.is_shared)
        existing_data.share_group = update_dict.get("share_group", existing_data.share_group)

        deploy_enabled = update_dict.get("enable_deploy", deploy_meta.is_deployed)
        deploy_meta.is_deployed = deploy_enabled

        if deploy_enabled:
            alphabet = string.ascii_letters + string.digits
            deploy_key = ''.join(secrets.choice(alphabet) for _ in range(32))
            deploy_meta.deploy_key = deploy_key

            logger.info(f"Generated new deploy key for workflow: {workflow_name}")
        else:
            deploy_meta.deploy_key = ""
            logger.info(f"Cleared deploy key for workflow: {workflow_name}")

        app_db.update(existing_data)
        app_db.update(deploy_meta)

        return {
            "message": "Workflow updated successfully",
            "workflow_name": existing_data.workflow_name,
            "deploy_key": deploy_meta.deploy_key if deploy_meta.is_deployed else None,
        }

    except Exception as e:
        logger.error(f"Failed to update workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update workflow: {str(e)}")




@router.delete("/delete/{workflow_name}")
async def delete_workflow(request: Request, user_id, workflow_name: str):
    """
    특정 workflow를 삭제합니다.
    """
    try:
        validate_superuser(request)
        app_db = get_db_manager(request)

        existing_data = app_db.find_by_condition(
            WorkflowMeta,
            {
                "user_id": user_id,
                "workflow_name": workflow_name,
            },
            ignore_columns=['workflow_data'],
            limit=1
        )

        app_db.delete(WorkflowMeta, existing_data[0].id if existing_data else None)
        app_db.delete_by_condition(DeployMeta, {
            "user_id": user_id,
            "workflow_id": existing_data[0].workflow_id,
            "workflow_name": workflow_name,
        })

        downloads_path = os.path.join(os.getcwd(), "downloads")
        download_path_id = os.path.join(downloads_path, user_id)
        filename = f"{workflow_name}.json"
        file_path = os.path.join(download_path_id, filename)

        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"Workflow '{workflow_name}' not found")

        os.remove(file_path)

        logger.info(f"Workflow deleted successfully: {filename}")
        return JSONResponse(content={
            "success": True,
            "message": f"Workflow '{workflow_name}' deleted successfully"
        })

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Workflow '{workflow_name}' not found")
    except Exception as e:
        logger.error(f"Error deleting workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete workflow: {str(e)}")

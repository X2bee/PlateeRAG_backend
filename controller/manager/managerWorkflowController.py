import logging
import json
import string
import secrets

from datetime import datetime
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse
from controller.helper.singletonHelper import get_db_manager
from controller.helper.utils.data_parsers import parse_input_data
from controller.helper.controllerHelper import require_admin_access
from service.database.logger_helper import create_logger
from service.database.models.user import User
from service.database.models.executor import ExecutionIO
from service.database.models.workflow import WorkflowMeta
from service.database.models.deploy import DeployMeta

logger = logging.getLogger("manager-workflow-controller")
router = APIRouter(prefix="/workflow", tags=["Manager"])

def get_manager_groups(app_db, manager_id):
    manager_account = app_db.find_by_condition(User, {"id": manager_id})
    manager_groups = manager_account[0].groups if manager_account else []
    admin_groups = [group for group in manager_groups if group.endswith("__admin__")]
    admin_groups = [group.replace("__admin__", "") for group in admin_groups]
    return list(set(admin_groups))

def get_manager_accessible_workflows(app_db, manager_id):
    admin_groups = get_manager_groups(app_db, manager_id)
    workflow_ids = []
    my_workflow_metas = app_db.find_by_condition(WorkflowMeta, {"user_id": manager_id}, select_columns=["workflow_id"])
    workflow_ids.extend([meta.workflow_id for meta in my_workflow_metas])

    if admin_groups:
        workflow_metas = app_db.find_by_condition(WorkflowMeta, {"share_group__in__": admin_groups}, select_columns=["workflow_id"])
        workflow_ids.extend([meta.workflow_id for meta in workflow_metas])

    all_users = app_db.find_by_condition(
        User,
        {"user_type__in__": ['admin', 'standard']},
        limit=10000
    )

    filtered_users = []
    for user in all_users:
        user_groups = user.groups if user.groups else []
        user_normal_groups = [g for g in user_groups if not g.endswith("__admin__")]
        user_groups_set = set(user_normal_groups)

        if user_groups_set and user_groups_set.issubset(set(admin_groups)):
            filtered_users.append(user)

    filtered_user_ids = [user.id for user in filtered_users]
    if filtered_user_ids:
        user_workflow_metas = app_db.find_by_condition(WorkflowMeta, {"user_id__in__": filtered_user_ids}, select_columns=["workflow_id"])
        workflow_ids.extend([meta.workflow_id for meta in user_workflow_metas])

    return list(set(workflow_ids))

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


@router.get("/manager-io-logs")
async def get_io_logs_by_id(request: Request, user_id = None, workflow_name: str = None, workflow_id: str = None):
    """
    매니저용 ExecutionIO 로그를 반환합니다. user_id가 없으면 모든 로그를 가져올 수 있습니다.
    """
    user_session = require_admin_access(request)
    if not user_session:
        raise HTTPException(
            status_code=403,
            detail="Admin access required"
        )

    app_db = get_db_manager(request)
    # 매니저는 자신의 user_id를 사용
    manager_user_id = request.state.user_id if hasattr(request.state, 'user_id') else None
    backend_log = create_logger(app_db, manager_user_id, request)

    try:
        conditions = {}
        if user_id:
            conditions['user_id'] = user_id
        if workflow_name:
            conditions['workflow_name'] = workflow_name
        if workflow_id:
            conditions['workflow_id'] = workflow_id

        if conditions:
            # find_by_condition을 사용하여 ExecutionIO와 users 테이블 조인
            result = app_db.find_by_condition(
                ExecutionIO,
                conditions,
                limit=1000000,
                offset=0,
                orderby="updated_at",
                orderby_asc=True,
                return_list=True,
                join_user=True
            )
        else:
            raise HTTPException(
                status_code=400,
                detail="At least one filter parameter (user_id, workflow_name, workflow_id) must be provided"
            )

        if not result:
            backend_log.info("No IO logs found for given conditions",
                           metadata={"conditions": conditions})
            logger.info("No IO logs found")
            return JSONResponse(content={
                "io_logs": [],
                "message": "No IO logs found"
            })

        io_logs = []
        for idx, row in enumerate(result):
            # input_data 파싱 (row는 이제 딕셔너리 형태)
            raw_input_data = json.loads(row['input_data']).get('result', None) if row['input_data'] else None
            parsed_input_data = parse_input_data(raw_input_data) if raw_input_data else None

            log_entry = {
                "log_id": idx + 1,
                "io_id": row['id'],
                "user_id": row['user_id'],
                "username": row['username'],
                "full_name": row['full_name'],
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

        backend_log.success("Successfully retrieved IO logs",
                          metadata={"conditions": conditions, "log_count": len(io_logs)})
        return JSONResponse(content=response_data)

    except Exception as e:
        backend_log.error("Error fetching IO logs", exception=e,
                         metadata={"conditions": {"user_id": user_id, "workflow_name": workflow_name, "workflow_id": workflow_id}})
        logger.error(f"Error fetching IO logs: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        ) from e

@router.get("/all-io-logs")
async def get_all_workflows_by_id(request: Request, page: int = 1, page_size: int = 250, user_id = None, workflow_id: str = None, workflow_name: str = None):
    """
    매니저용 모든 ExecutionIO 로그를 페이지네이션과 함께 반환합니다.
    """
    user_session = require_admin_access(request)
    if not user_session:
        raise HTTPException(
            status_code=403,
            detail="Admin access required"
        )
    app_db = get_db_manager(request)
    manager_id = user_session['user_id']
    backend_log = create_logger(app_db, manager_id, request)

    db_type = app_db.config_db_manager.db_type
    workflow_ids = get_manager_accessible_workflows(app_db, manager_id)
    if workflow_id:
        if workflow_id in workflow_ids:
            logger.info(f"Filtering logs for workflow_id: {workflow_id}")
        else:
            raise HTTPException(
                status_code=403,
                detail="You do not have access to this workflow"
            )
    try:
        # 페이지 번호 검증
        if page < 1:
            page = 1
        if page_size < 1 or page_size > 1000:
            page_size = 250

        offset = (page - 1) * page_size

        app_db = get_db_manager(request)

        conditions = {
            "workflow_id__in__": workflow_ids
        }
        if user_id:
            conditions['user_id'] = user_id
        if workflow_name:
            conditions['workflow_name'] = workflow_name

        io_logs_result = app_db.find_by_condition(
            ExecutionIO,
            conditions,
            limit=page_size,
            offset=offset,
            orderby="updated_at",
            orderby_asc=False,
            return_list=True,
            join_user=True
        )

        # 딕셔너리 형태의 결과를 처리
        processed_logs = []
        for log in io_logs_result:
            log_dict = dict(log)
            log_dict.update({
                "input_data": extract_result_from_json(log_dict["input_data"]),
                "output_data": extract_result_from_json(log_dict["output_data"])
            })
            processed_logs.append(log_dict)

        backend_log.success("Successfully retrieved all IO logs",
                          metadata={"page": page, "page_size": page_size, "log_count": len(processed_logs)})
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
        backend_log.error("Error fetching all IO logs", exception=e,
                         metadata={"page": page, "page_size": page_size, "user_id": user_id})
        logger.error("Error fetching all IO logs: %s", str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        ) from e

@router.get("/all-list")
async def get_all_workflows(request: Request, page: int = 1, page_size: int = 250, user_id = None):
    """
    매니저용 워크플로우 목록을 반환합니다.
    매니저는 자신이 만든 워크플로우와 자신이 admin으로 속한 그룹에서 공유된 워크플로우만 볼 수 있습니다.
    """
    user_session = require_admin_access(request)
    if not user_session:
        raise HTTPException(
            status_code=403,
            detail="Admin access required"
        )

    app_db = get_db_manager(request)
    manager_id = user_session['user_id']
    backend_log = create_logger(app_db, manager_id, request)

    try:
        # 페이지 번호 검증
        if page < 1:
            page = 1
        if page_size < 1 or page_size > 1000:
            page_size = 250

        offset = (page - 1) * page_size

        workflow_ids = get_manager_accessible_workflows(app_db, manager_id)

        if not workflow_ids:
            return {
                "workflows": [],
                "pagination": {
                    "page": page,
                    "page_size": page_size,
                    "offset": offset,
                    "total_returned": 0
                }
            }

        # workflow_id 리스트를 사용하여 조인 쿼리 실행
        db_type = app_db.config_db_manager.db_type

        if db_type == "postgresql":
            placeholders = ', '.join(['%s'] * len(workflow_ids))
            query = f"""
                SELECT
                    wm.id, wm.created_at, wm.updated_at,
                    wm.user_id, wm.workflow_id, wm.workflow_name,
                    wm.node_count, wm.edge_count, wm.has_startnode, wm.has_endnode,
                    wm.is_completed, wm.metadata, wm.is_shared, wm.share_group, wm.share_permissions,
                    u.full_name, u.username,
                    dm.is_deployed, dm.deploy_key, dm.is_accepted, dm.inquire_deploy
                FROM workflow_meta wm
                LEFT JOIN users u ON wm.user_id = u.id
                LEFT JOIN deploy_meta dm ON wm.workflow_id = dm.workflow_id AND wm.workflow_name = dm.workflow_name AND wm.user_id = dm.user_id
                WHERE wm.workflow_id IN ({placeholders})
                ORDER BY wm.created_at DESC
                LIMIT %s OFFSET %s
            """
            params = tuple(workflow_ids) + (page_size, offset)
        else:
            placeholders = ', '.join(['?'] * len(workflow_ids))
            query = f"""
                SELECT
                    wm.id, wm.created_at, wm.updated_at,
                    wm.user_id, wm.workflow_id, wm.workflow_name,
                    wm.node_count, wm.edge_count, wm.has_startnode, wm.has_endnode,
                    wm.is_completed, wm.metadata, wm.is_shared, wm.share_group, wm.share_permissions,
                    u.full_name, u.username,
                    dm.is_deployed, dm.deploy_key, dm.is_accepted, dm.inquire_deploy
                FROM workflow_meta wm
                LEFT JOIN users u ON wm.user_id = u.id
                LEFT JOIN deploy_meta dm ON wm.workflow_id = dm.workflow_id AND wm.workflow_name = dm.workflow_name AND wm.user_id = dm.user_id
                WHERE wm.workflow_id IN ({placeholders})
                ORDER BY wm.created_at DESC
                LIMIT ? OFFSET ?
            """
            params = tuple(workflow_ids) + (page_size, offset)

        all_workflows = app_db.config_db_manager.execute_query(query, params)

        # id 중복 제거 - id를 기준으로 중복된 항목은 첫 번째 것만 유지
        seen_ids = set()
        unique_workflows = []
        for workflow in all_workflows:
            workflow_id = workflow['id']
            if workflow_id not in seen_ids:
                seen_ids.add(workflow_id)
                unique_workflows.append(workflow)

        backend_log.success("Successfully retrieved workflow list",
                          metadata={"page": page, "page_size": page_size, "workflow_count": len(unique_workflows)})
        return {
            "workflows": unique_workflows,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "offset": offset,
                "total_returned": len(unique_workflows)
            }
        }
    except Exception as e:
        backend_log.error("Error fetching manager workflows", exception=e,
                         metadata={"page": page, "page_size": page_size, "user_id": user_id})
        logger.error("Error fetching manager workflows: %s", str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        ) from e


@router.post("/update/{workflow_name}")
async def update_workflow(request: Request, workflow_name: str, update_dict: dict):
    user_session = require_admin_access(request)
    if not user_session:
        raise HTTPException(
            status_code=403,
            detail="Admin access required"
        )
    manager_id = user_session['user_id']
    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, manager_id, request)

    manager_accessible_workflows = get_manager_accessible_workflows(app_db, manager_id)
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

        if existing_data[0].workflow_id not in manager_accessible_workflows:
            raise HTTPException(
                status_code=403,
                detail="You do not have permission to modify this workflow"
            )

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

        deploy_meta.is_accepted = update_dict.get("is_accepted", deploy_meta.is_accepted)
        deploy_meta.inquire_deploy = update_dict.get("inquire_deploy", deploy_meta.inquire_deploy)

        if deploy_enabled:
            alphabet = string.ascii_letters + string.digits
            deploy_key = ''.join(secrets.choice(alphabet) for _ in range(32))
            deploy_meta.deploy_key = deploy_key
            deploy_meta.inquire_deploy = False

            logger.info(f"Generated new deploy key for workflow: {workflow_name}")
        else:
            deploy_meta.deploy_key = ""
            logger.info(f"Cleared deploy key for workflow: {workflow_name}")

        app_db.update(existing_data)
        app_db.update(deploy_meta)

        backend_log.success(f"Successfully updated workflow: {workflow_name}",
                          metadata={"workflow_name": workflow_name, "user_id": user_id, "is_deployed": deploy_meta.is_deployed})
        return {
            "message": "Workflow updated successfully",
            "workflow_name": existing_data.workflow_name,
            "deploy_key": deploy_meta.deploy_key if deploy_meta.is_deployed else None,
        }

    except Exception as e:
        backend_log.error(f"Failed to update workflow: {workflow_name}", exception=e,
                         metadata={"workflow_name": workflow_name, "user_id": user_id})
        logger.error(f"Failed to update workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update workflow: {str(e)}")

@router.delete("/delete/{workflow_name}")
async def delete_workflow(request: Request, user_id, workflow_name: str):
    """
    특정 workflow를 삭제합니다.
    """
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required as a query parameter")
    try:
        user_session = require_admin_access(request)
        if not user_session:
            raise HTTPException(
                status_code=403,
                detail="Admin access required"
            )
        manager_id = user_session['user_id']
        app_db = get_db_manager(request)
        backend_log = create_logger(app_db, manager_id, request)

        manager_accessible_workflows = get_manager_accessible_workflows(app_db, manager_id)

        existing_data = app_db.find_by_condition(
            WorkflowMeta,
            {
                "user_id": user_id,
                "workflow_name": workflow_name,
            },
            ignore_columns=['workflow_data'],
            limit=1
        )
        if not existing_data:
            raise HTTPException(status_code=404, detail="Workflow not found")
        if existing_data[0].workflow_id not in manager_accessible_workflows:
            raise HTTPException(
                status_code=403,
                detail="You do not have permission to modify this workflow"
            )

        app_db.delete(WorkflowMeta, existing_data[0].id if existing_data else None)
        app_db.delete_by_condition(DeployMeta, {
            "user_id": user_id,
            "workflow_id": existing_data[0].workflow_id,
            "workflow_name": workflow_name,
        })

        backend_log.success(f"Successfully deleted workflow: {workflow_name}",
                          metadata={"workflow_name": workflow_name, "user_id": user_id})
        logger.info(f"Workflow deleted successfully: {workflow_name}")
        return JSONResponse(content={
            "success": True,
            "message": f"Workflow '{workflow_name}' deleted successfully"
        })

    except FileNotFoundError:
        backend_log.warn(f"Workflow not found for deletion: {workflow_name}",
                        metadata={"workflow_name": workflow_name, "user_id": user_id})
        raise HTTPException(status_code=404, detail=f"Workflow '{workflow_name}' not found")
    except Exception as e:
        backend_log.error(f"Error deleting workflow: {workflow_name}", exception=e,
                         metadata={"workflow_name": workflow_name, "user_id": user_id})
        logger.error(f"Error deleting workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete workflow: {str(e)}")

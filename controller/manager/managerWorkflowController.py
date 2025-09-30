import logging
import json
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

logger = logging.getLogger("manager-workflow-controller")
router = APIRouter(prefix="/workflow", tags=["Manager"])

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
async def get_all_workflows_by_id(request: Request, page: int = 1, page_size: int = 250, user_id = None):
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
    manager_account = app_db.find_by_condition(User, {"id": manager_id})
    manager_groups = manager_account[0].groups
    admin_groups = [group for group in manager_groups if group.endswith("__admin__")]
    admin_groups = [group.replace("__admin__", "") for group in admin_groups]

    workflow_ids = []
    my_workflow_metas = app_db.find_by_condition(WorkflowMeta, {"user_id": manager_id}, select_columns=["workflow_id"])
    workflow_ids.extend([meta.workflow_id for meta in my_workflow_metas])

    if admin_groups:
        workflow_metas = app_db.find_by_condition(WorkflowMeta, {"share_group__in__": admin_groups}, select_columns=["workflow_id"])
        workflow_ids.extend([meta.workflow_id for meta in workflow_metas])

    workflow_ids = list(set(workflow_ids))
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

    try:
        # 페이지 번호 검증
        if page < 1:
            page = 1
        if page_size < 1 or page_size > 1000:
            page_size = 250

        offset = (page - 1) * page_size

        app_db = get_db_manager(request)
        manager_id = user_session['user_id']
        
        # 매니저의 그룹 정보 가져오기
        manager_account = app_db.find_by_condition(User, {"id": manager_id})
        manager_groups = manager_account[0].groups if manager_account else []
        admin_groups = [group for group in manager_groups if group.endswith("__admin__")]
        admin_groups = [group.replace("__admin__", "") for group in admin_groups]

        # 워크플로우 ID 수집: 자신이 만든 것 + admin 그룹에서 공유된 것
        workflow_ids = []
        
        # 1. 자신이 만든 워크플로우
        my_workflow_metas = app_db.find_by_condition(
            WorkflowMeta, 
            {"user_id": manager_id if not user_id else user_id}, 
            select_columns=["workflow_id"]
        )
        workflow_ids.extend([meta.workflow_id for meta in my_workflow_metas])

        # 2. admin 그룹에서 공유된 워크플로우 (user_id 필터가 없을 때만)
        if admin_groups and not user_id:
            shared_workflow_metas = app_db.find_by_condition(
                WorkflowMeta, 
                {"share_group__in__": admin_groups}, 
                select_columns=["workflow_id"]
            )
            workflow_ids.extend([meta.workflow_id for meta in shared_workflow_metas])

        # 중복 제거
        workflow_ids = list(set(workflow_ids))

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
                LEFT JOIN deploy_meta dm ON wm.workflow_id = dm.workflow_id
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
                LEFT JOIN deploy_meta dm ON wm.workflow_id = dm.workflow_id
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
        logger.error("Error fetching manager workflows: %s", str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        ) from e

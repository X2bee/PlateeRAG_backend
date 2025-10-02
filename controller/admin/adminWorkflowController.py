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
from controller.helper.utils.data_parsers import parse_input_data
from service.database.logger_helper import create_logger
from controller.admin.adminHelper import get_manager_groups, get_manager_accessible_users, manager_section_access, get_manager_accessible_workflows_ids
from service.database.models.executor import ExecutionIO
from service.database.models.workflow import WorkflowMeta
from service.database.models.deploy import DeployMeta
from service.database.models.user import User
from controller.helper.utils.data_parsers import safe_round_float
from service.database.models.performance import NodePerformance

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
async def get_io_logs_by_id(request: Request, user_id: int = None, workflow_name: str = None, workflow_id: str = None):
    """
    관리자용 ExecutionIO 로그를 반환합니다. user_id가 없으면 모든 로그를 가져올 수 있습니다.
    """
    val_superuser = await validate_superuser(request)
    if val_superuser.get("superuser") is not True:
        raise HTTPException(
            status_code=403,
            detail="Admin privileges required"
        )

    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, val_superuser.get("user_id"), request)
    section_access = manager_section_access(app_db, val_superuser.get("user_id"), ["workflow-monitoring", "chat-monitoring", "workflow-management"])
    if not section_access:
        backend_log.warn(f"User {val_superuser.get('user_id')} attempted to access IO logs without permission")
        raise HTTPException(
            status_code=403,
            detail="Group permissions access required"
        )
    try:
        if user_id:
            if val_superuser.get("user_type") != "superuser":
                manager_accessible_users = get_manager_accessible_users(app_db, val_superuser.get("user_id"))
                if user_id not in [user.id for user in manager_accessible_users]:
                    raise HTTPException(
                        status_code=403,
                        detail="You do not have permission to access logs for this user"
                    )
            conditions = {'user_id': user_id}
        else:
            if val_superuser.get("user_type") != "superuser":
                manager_accessible_users = get_manager_accessible_users(app_db, val_superuser.get("user_id"))
                accessible_user_ids = [user.id for user in manager_accessible_users]
                logger.info(f"Manager accessible users: {accessible_user_ids}")
                conditions = {"user_id__in__": accessible_user_ids}
            else:
                conditions = {}

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
async def get_all_workflows_by_id(request: Request, page: int = 1, page_size: int = 250, user_id: int = None, workflow_id: str = None, workflow_name: str = None):
    val_superuser = await validate_superuser(request)
    if val_superuser.get("superuser") is not True:
        raise HTTPException(
            status_code=403,
            detail="Admin privileges required"
        )
    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, val_superuser.get("user_id"), request)
    section_access = manager_section_access(app_db, val_superuser.get("user_id"), ["workflow-monitoring", "chat-monitoring", "workflow-management"])

    if not section_access:
        backend_log.warn(f"User {val_superuser.get('user_id')} attempted to access IO logs without permission")
        raise HTTPException(
            status_code=403,
            detail="Group permissions access required"
        )

    try:
        if page < 1:
            page = 1
        if page_size < 1 or page_size > 1000:
            page_size = 250

        offset = (page - 1) * page_size

        if user_id:
            if val_superuser.get("user_type") != "superuser":
                manager_accessible_users = get_manager_accessible_users(app_db, val_superuser.get("user_id"))
                if user_id not in [user.id for user in manager_accessible_users]:
                    raise HTTPException(
                        status_code=403,
                        detail="You do not have permission to access logs for this user"
                    )
            conditions = {'user_id': user_id}
        else:
            if val_superuser.get("user_type") != "superuser":
                manager_accessible_users = get_manager_accessible_users(app_db, val_superuser.get("user_id"))
                accessible_user_ids = [user.id for user in manager_accessible_users]
                logger.info(f"Manager accessible users: {accessible_user_ids}")
                conditions = {"user_id__in__": accessible_user_ids}
            else:
                conditions = {}
        if workflow_id:
            conditions['workflow_id'] = workflow_id
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


@router.get("/all-io-logs/tester")
async def get_workflow_io_logs_for_tester(request: Request, user_id: int = None, workflow_name: str = None):
    """
    특정 워크플로우의 ExecutionIO 로그를 interaction_batch_id별로 그룹화하여 반환합니다.
    """
    val_superuser = await validate_superuser(request)
    if val_superuser.get("superuser") is not True:
        raise HTTPException(
            status_code=403,
            detail="Admin privileges required"
        )

    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, val_superuser.get("user_id"), request)
    section_access = manager_section_access(app_db, val_superuser.get("user_id"), ["workflow-monitoring"])
    if not section_access:
        backend_log.warn(f"User {val_superuser.get('user_id')} attempted to access IO logs without permission")
        raise HTTPException(
            status_code=403,
            detail="Group permissions access required"
        )

    if val_superuser.get("user_type") != "superuser":
        manager_accessible_users = get_manager_accessible_users(app_db, val_superuser.get("user_id"))
        accessible_user_ids = [user.id for user in manager_accessible_users]
        if user_id not in accessible_user_ids:
            logger.warning(f"User {val_superuser.get('user_id')} attempted to access IO logs for user {user_id} without permission. Accessible users: {accessible_user_ids}")
            raise HTTPException(
                status_code=403,
                detail="You do not have permission to access logs for this user"
            )

    try:
        backend_log.info("Retrieving workflow tester IO logs",
                        metadata={"workflow_name": workflow_name})

        result = app_db.find_by_condition(
            ExecutionIO,
            {
                "user_id": user_id,
                "workflow_name": workflow_name,
                "test_mode": True
            },
            limit=1000000,
            orderby="updated_at",
            orderby_asc=True,
            return_list=True
        )

        if not result:
            backend_log.info("No tester IO logs found",
                           metadata={"workflow_name": workflow_name})
            logger.info(f"No performance data found for workflow: {workflow_name}")
            return JSONResponse(content={
                "workflow_name": workflow_name,
                "response_data_list": [],
                "message": "No in_out_logs data found for this workflow"
            })

        # interaction_batch_id별로 그룹화
        tester_groups = {}

        for idx, row in enumerate(result):
            interaction_id = row['interaction_id']

            # interaction_id에서 마지막 숫자를 제외한 배치 ID 추출
            parts = interaction_id.split('____')
            if len(parts) >= 4:
                interaction_batch_id = '____'.join(parts[:-1])
            else:
                interaction_batch_id = interaction_id

            if interaction_batch_id not in tester_groups:
                tester_groups[interaction_batch_id] = []

            # input_data 파싱
            raw_input_data = json.loads(row['input_data']).get('result', None) if row['input_data'] else None
            parsed_input_data = parse_input_data(raw_input_data) if raw_input_data else None

            # interaction_id에서 마지막 번호를 추출하여 log_id로 사용
            parts = interaction_id.split('____')
            if len(parts) >= 4 and parts[-1].isdigit():
                log_id = int(parts[-1])
            else:
                log_id = len(tester_groups[interaction_batch_id]) + 1

            log_entry = {
                "log_id": log_id,
                "interaction_id": row['interaction_id'],
                "workflow_name": row['workflow_name'],
                "workflow_id": row['workflow_id'],
                "input_data": parsed_input_data,
                "output_data": json.loads(row['output_data']).get('result', None) if row['output_data'] else None,
                "expected_output": row['expected_output'],
                "llm_eval_score": row['llm_eval_score'],
                "updated_at": row['updated_at'].isoformat() if isinstance(row['updated_at'], datetime) else row['updated_at']
            }
            tester_groups[interaction_batch_id].append(log_entry)

        # 각 테스터 그룹을 response_data 형태로 변환
        response_data_list = []
        for interaction_batch_id, performance_stats in tester_groups.items():
            response_data = {
                "workflow_name": workflow_name,
                "interaction_batch_id": interaction_batch_id,
                "in_out_logs": performance_stats,
                "message": "In/Out logs retrieved successfully"
            }
            response_data_list.append(response_data)

        final_response = {
            "workflow_name": workflow_name,
            "response_data_list": response_data_list,
            "message": f"In/Out logs retrieved successfully for {len(response_data_list)} tester groups"
        }

        backend_log.success("Successfully retrieved workflow tester IO logs",
                          metadata={"workflow_name": workflow_name,
                                  "tester_groups": len(response_data_list),
                                  "total_logs": len(result)})

        logger.info(f"Performance stats retrieved for workflow: {workflow_name}, {len(response_data_list)} tester groups")
        return JSONResponse(content=final_response)

    except Exception as e:
        backend_log.error("Failed to retrieve workflow tester IO logs", exception=e,
                         metadata={"workflow_name": workflow_name})
        logger.error(f"Error retrieving workflow performance: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve performance data: {str(e)}")

@router.get("/all-list")
async def get_all_workflows(request: Request, page: int = 1, page_size: int = 250, user_id = None):
    val_superuser = await validate_superuser(request)
    if val_superuser.get("superuser") is not True:
        raise HTTPException(
            status_code=403,
            detail="Admin privileges required"
        )
    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, val_superuser.get("user_id"), request)
    section_access = manager_section_access(app_db, val_superuser.get("user_id"), ["workflow-monitoring", "workflow-management"])
    if not section_access:
        backend_log.warn(f"User {val_superuser.get('user_id')} attempted to access IO logs without permission")
        raise HTTPException(
            status_code=403,
            detail="Group permissions access required"
        )
    db_type = app_db.config_db_manager.db_type
    try:
        # 페이지 번호 검증
        if page < 1:
            page = 1
        if page_size < 1 or page_size > 1000:
            page_size = 250

        offset = (page - 1) * page_size

        if user_id:
            if val_superuser.get("user_type") != "superuser":
                manager_accessible_users = get_manager_accessible_users(app_db, val_superuser.get("user_id"))
                if user_id not in [user.id for user in manager_accessible_users]:
                    raise HTTPException(
                        status_code=403,
                        detail="You do not have permission to access logs for this user"
                    )
            query = """
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
                WHERE wm.user_id = %s
                ORDER BY wm.created_at DESC
                LIMIT %s OFFSET %s
            """
            if db_type != "postgresql":
                query = query.replace("%s", "?")
            all_workflows = app_db.config_db_manager.execute_query(query, (user_id, page_size, offset))
        else:
            query = """
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
                ORDER BY wm.created_at DESC
                LIMIT %s OFFSET %s
            """
            if val_superuser.get("user_type") != "superuser":
                accessible_workflow_ids = get_manager_accessible_workflows_ids(app_db, val_superuser.get("user_id"))
                manager_accessible_users = get_manager_accessible_users(app_db, val_superuser.get("user_id"))
                accessible_user_ids = [user.id for user in manager_accessible_users]

                if db_type == "postgresql":
                    workflow_placeholders = ', '.join(['%s'] * len(accessible_workflow_ids))
                    user_placeholders = ', '.join(['%s'] * len(accessible_user_ids))
                else:
                    workflow_placeholders = ', '.join(['?'] * len(accessible_workflow_ids))
                    user_placeholders = ', '.join(['?'] * len(accessible_user_ids))
                query = query.replace("ORDER BY", f"WHERE wm.workflow_id IN ({workflow_placeholders}) AND wm.user_id IN ({user_placeholders}) ORDER BY")
                params = accessible_workflow_ids + accessible_user_ids + [page_size, offset]
                all_workflows = app_db.config_db_manager.execute_query(query, params)
            else:
                if db_type != "postgresql":
                    query = query.replace("%s", "?")
                all_workflows = app_db.config_db_manager.execute_query(query, (page_size, offset))

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
        logger.error("Error fetching all IO logs: %s", str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        ) from e

@router.post("/update/{workflow_name}")
async def update_workflow(request: Request, workflow_name: str, update_dict: dict):
    val_superuser = await validate_superuser(request)
    if val_superuser.get("superuser") is not True:
        raise HTTPException(
            status_code=403,
            detail="Admin privileges required"
        )
    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, val_superuser.get("user_id"), request)
    section_access = manager_section_access(app_db, val_superuser.get("user_id"), ["workflow-management"])
    if not section_access:
        backend_log.warn(f"User {val_superuser.get('user_id')} attempted to access IO logs without permission")
        raise HTTPException(
            status_code=403,
            detail="Group permissions access required"
        )

    user_id = update_dict.get("user_id")
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required in the update data")
    if val_superuser.get("user_type") != "superuser":
        manager_accessible_users = get_manager_accessible_users(app_db, val_superuser.get("user_id"))
        if user_id not in [user.id for user in manager_accessible_users]:
            raise HTTPException(
                status_code=403,
                detail="You do not have permission to update workflows for this user"
            )

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
    val_superuser = await validate_superuser(request)
    if val_superuser.get("superuser") is not True:
        raise HTTPException(
            status_code=403,
            detail="Admin privileges required"
        )
    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, val_superuser.get("user_id"), request)
    section_access = manager_section_access(app_db, val_superuser.get("user_id"), ["workflow-management"])
    if not section_access:
        backend_log.warn(f"User {val_superuser.get('user_id')} attempted to access IO logs without permission")
        raise HTTPException(
            status_code=403,
            detail="Group permissions access required"
        )
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required in the update data")
    if val_superuser.get("user_type") != "superuser":
        manager_accessible_users = get_manager_accessible_users(app_db, val_superuser.get("user_id"))
        if user_id not in [user.id for user in manager_accessible_users]:
            raise HTTPException(
                status_code=403,
                detail="You do not have permission to update workflows for this user"
            )

    try:
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

        logger.info(f"Workflow deleted successfully: {workflow_name}")
        return JSONResponse(content={
            "success": True,
            "message": f"Workflow '{workflow_name}' deleted successfully"
        })

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Workflow '{workflow_name}' not found")
    except Exception as e:
        logger.error(f"Error deleting workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete workflow: {str(e)}")

@router.get("/performance")
async def get_workflow_performance(request: Request, user_id: int, workflow_name: str, workflow_id: str):
    """
    특정 워크플로우의 성능 통계를 반환합니다.
    node_id와 node_name별로 평균 성능 지표를 계산합니다.
    """
    val_superuser = await validate_superuser(request)
    if val_superuser.get("superuser") is not True:
        raise HTTPException(
            status_code=403,
            detail="Admin privileges required"
        )
    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, val_superuser.get("user_id"), request)
    section_access = manager_section_access(app_db, val_superuser.get("user_id"), ["workflow-monitoring"])
    if not section_access:
        backend_log.warn(f"User {val_superuser.get('user_id')} attempted to access IO logs without permission")
        raise HTTPException(
            status_code=403,
            detail="Group permissions access required"
        )
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required in the update data")
    if val_superuser.get("user_type") != "superuser":
        manager_accessible_users = get_manager_accessible_users(app_db, val_superuser.get("user_id"))
        if user_id not in [user.id for user in manager_accessible_users]:
            raise HTTPException(
                status_code=403,
                detail="You do not have permission to update workflows for this user"
            )
    try:
        backend_log.info("Retrieving workflow performance statistics",
                        metadata={"workflow_name": workflow_name, "workflow_id": workflow_id})

        # SQL 쿼리 작성
        query = """
        SELECT
            node_id,
            node_name,
            AVG(processing_time_ms) as avg_processing_time_ms,
            AVG(cpu_usage_percent) as avg_cpu_usage_percent,
            AVG(ram_usage_mb) as avg_ram_usage_mb,
            AVG(CASE WHEN gpu_usage_percent IS NOT NULL THEN gpu_usage_percent END) as avg_gpu_usage_percent,
            AVG(CASE WHEN gpu_memory_mb IS NOT NULL THEN gpu_memory_mb END) as avg_gpu_memory_mb,
            COUNT(*) as execution_count,
            COUNT(CASE WHEN gpu_usage_percent IS NOT NULL THEN 1 END) as gpu_execution_count
        FROM node_performance
        WHERE workflow_name = %s AND workflow_id = %s AND user_id = %s
        GROUP BY node_id, node_name
        ORDER BY node_id
        """

        # SQLite인 경우 파라미터 플레이스홀더 변경
        if app_db.config_db_manager.db_type == "sqlite":
            query = query.replace("%s", "?")

        # 쿼리 실행
        result = app_db.config_db_manager.execute_query(query, (workflow_name, workflow_id, user_id))

        if not result:
            backend_log.info("No performance data found",
                           metadata={"workflow_name": workflow_name, "workflow_id": workflow_id})
            logger.info(f"No performance data found for workflow: {workflow_name} ({workflow_id})")
            return JSONResponse(content={
                "workflow_name": workflow_name,
                "workflow_id": workflow_id,
                "performance_stats": [],
                "message": "No performance data found for this workflow"
            })

        # 결과 포맷팅
        performance_stats = []
        for row in result:
            stats = {
                "node_id": row['node_id'],
                "node_name": row['node_name'],
                "avg_processing_time_ms": safe_round_float(row['avg_processing_time_ms']) if row['avg_processing_time_ms'] else 0.0,
                "avg_cpu_usage_percent": safe_round_float(row['avg_cpu_usage_percent']) if row['avg_cpu_usage_percent'] else 0.0,
                "avg_ram_usage_mb": safe_round_float(row['avg_ram_usage_mb']) if row['avg_ram_usage_mb'] else 0.0,
                "avg_gpu_usage_percent": safe_round_float(row['avg_gpu_usage_percent']) if row['avg_gpu_usage_percent'] else None,
                "avg_gpu_memory_mb": safe_round_float(row['avg_gpu_memory_mb']) if row['avg_gpu_memory_mb'] else None,
                "execution_count": int(row['execution_count']) if row['execution_count'] else 0,
                "gpu_execution_count": int(row['gpu_execution_count']) if row['gpu_execution_count'] else 0
            }
            performance_stats.append(stats)

        # 전체 워크플로우 통계 계산
        total_executions = sum(stat['execution_count'] for stat in performance_stats)
        avg_total_processing_time = sum(float(stat['avg_processing_time_ms']) * stat['execution_count'] for stat in performance_stats) / total_executions if total_executions > 0 else 0.0
        avg_total_cpu_usage = sum(float(stat['avg_cpu_usage_percent']) * stat['execution_count'] for stat in performance_stats) / total_executions if total_executions > 0 else 0.0
        avg_total_ram_usage = sum(float(stat['avg_ram_usage_mb']) * stat['execution_count'] for stat in performance_stats) / total_executions if total_executions > 0 else 0.0

        # GPU 통계
        gpu_stats = None
        total_gpu_executions = sum(stat['gpu_execution_count'] for stat in performance_stats)
        if total_gpu_executions > 0:
            gpu_usage_sum = sum(float(stat['avg_gpu_usage_percent']) * stat['gpu_execution_count'] for stat in performance_stats if stat['avg_gpu_usage_percent'] is not None)
            gpu_memory_sum = sum(float(stat['avg_gpu_memory_mb']) * stat['gpu_execution_count'] for stat in performance_stats if stat['avg_gpu_memory_mb'] is not None)

            gpu_stats = {
                "avg_gpu_usage_percent": round(float(gpu_usage_sum / total_gpu_executions), 2) if total_gpu_executions > 0 else None,
                "avg_gpu_memory_mb": round(float(gpu_memory_sum / total_gpu_executions), 2) if total_gpu_executions > 0 else None,
                "gpu_execution_count": total_gpu_executions
            }

        response_data = {
            "workflow_name": workflow_name,
            "workflow_id": workflow_id,
            "summary": {
                "total_executions": total_executions,
                "avg_total_processing_time_ms": round(float(avg_total_processing_time), 2),
                "avg_total_cpu_usage_percent": round(float(avg_total_cpu_usage), 2),
                "avg_total_ram_usage_mb": round(float(avg_total_ram_usage), 2),
                "gpu_stats": gpu_stats
            },
            "performance_stats": performance_stats
        }

        backend_log.success("Successfully retrieved workflow performance statistics",
                          metadata={"workflow_name": workflow_name,
                                  "workflow_id": workflow_id,
                                  "total_executions": total_executions,
                                  "nodes_analyzed": len(performance_stats),
                                  "gpu_executions": total_gpu_executions,
                                  "avg_processing_time": round(float(avg_total_processing_time), 2)})

        logger.info(f"Performance stats retrieved for workflow: {workflow_name} ({workflow_id})")
        return JSONResponse(content=response_data)

    except Exception as e:
        backend_log.error("Failed to retrieve workflow performance statistics", exception=e,
                         metadata={"workflow_name": workflow_name, "workflow_id": workflow_id})
        logger.error(f"Error retrieving workflow performance: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve performance data: {str(e)}")

@router.delete("/performance")
async def delete_workflow_performance(request: Request, user_id: int, workflow_name: str, workflow_id: str):
    """
    특정 워크플로우의 성능 데이터를 삭제합니다.
    """
    val_superuser = await validate_superuser(request)
    if val_superuser.get("superuser") is not True:
        raise HTTPException(
            status_code=403,
            detail="Admin privileges required"
        )
    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, val_superuser.get("user_id"), request)
    section_access = manager_section_access(app_db, val_superuser.get("user_id"), ["workflow-monitoring"])
    if not section_access:
        backend_log.warn(f"User {val_superuser.get('user_id')} attempted to access IO logs without permission")
        raise HTTPException(
            status_code=403,
            detail="Group permissions access required"
        )
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required in the update data")
    if val_superuser.get("user_type") != "superuser":
        manager_accessible_users = get_manager_accessible_users(app_db, val_superuser.get("user_id"))
        if user_id not in [user.id for user in manager_accessible_users]:
            raise HTTPException(
                status_code=403,
                detail="You do not have permission to update workflows for this user"
            )
    try:
        backend_log.info("Starting workflow performance data deletion",
                        metadata={"workflow_name": workflow_name, "workflow_id": workflow_id})

        existing_data = app_db.find_by_condition(
            NodePerformance,
            {
                "user_id": user_id,
                "workflow_name": workflow_name,
                "workflow_id": workflow_id
            },
            limit=1000000
        )

        delete_count = len(existing_data) if existing_data else 0

        if delete_count == 0:
            backend_log.info("No performance data found to delete",
                           metadata={"workflow_name": workflow_name, "workflow_id": workflow_id})
            logger.info(f"No performance data found to delete for workflow: {workflow_name} ({workflow_id})")
            return JSONResponse(content={
                "workflow_name": workflow_name,
                "workflow_id": workflow_id,
                "deleted_count": 0,
                "message": "No performance data found to delete"
            })

        app_db.delete_by_condition(
            NodePerformance,
            {
                "user_id": user_id,
                "workflow_name": workflow_name,
                "workflow_id": workflow_id
            }
        )

        response_data = {
            "user_id": user_id,
            "workflow_name": workflow_name,
            "workflow_id": workflow_id,
            "deleted_count": delete_count,
            "message": f"Successfully deleted {delete_count} performance records"
        }

        backend_log.success("Successfully deleted workflow performance data",
                          metadata={"workflow_name": workflow_name,
                                  "workflow_id": workflow_id,
                                  "deleted_count": delete_count})

        logger.info(f"Deleted {delete_count} performance records for workflow: {workflow_name} ({workflow_id})")
        return JSONResponse(content=response_data)

    except Exception as e:
        backend_log.error("Failed to delete workflow performance data", exception=e,
                         metadata={"workflow_name": workflow_name,
                                 "workflow_id": workflow_id,
                                 "expected_delete_count": delete_count if 'delete_count' in locals() else 0})
        logger.error(f"Error deleting performance data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete performance data: {str(e)}")

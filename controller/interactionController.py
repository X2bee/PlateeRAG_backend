from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import os
import json
import logging
from datetime import datetime
from src.workflow_executor import WorkflowExecutor
from models.executor import ExecutionMeta

logger = logging.getLogger("interaction-controller")
router = APIRouter(prefix="/api/interaction", tags=["interaction"])

class WorkflowRequest(BaseModel):
    workflow_name: str
    workflow_id: str
    input_data: str = ""
    interaction_id: str = "default"

class WorkflowData(BaseModel):
    workflow_name: str
    workflow_id: str
    view: Dict[str, Any]
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    interaction_id: str = "default"

class SaveWorkflowRequest(BaseModel):
    workflow_id: str
    content: WorkflowData

@router.get("/list")
async def list_interaction(request: Request, interaction_id: str = None, workflow_id: str = None, limit: int = 100):
    """
    ExecutionMeta 정보들을 리스트 형태로 반환합니다.
    
    Args:
        interaction_id: 특정 상호작용 ID로 필터링 (선택적)
        workflow_id: 특정 워크플로우 ID로 필터링 (선택적)
        limit: 반환할 최대 레코드 수 (기본값: 100)
        
    Returns:
        ExecutionMeta 데이터 리스트
    """
    try:
        # 데이터베이스 매니저 가져오기
        if not hasattr(request.app.state, 'app_db') or not request.app.state.app_db:
            raise HTTPException(status_code=500, detail="Database connection not available")
        
        db_manager = request.app.state.app_db
        
        # SQL 쿼리 작성 - 조건에 따라 동적으로 구성
        where_conditions = []
        query_params = []
        
        if interaction_id:
            where_conditions.append("interaction_id = %s")
            query_params.append(interaction_id)
        
        if workflow_id:
            where_conditions.append("workflow_id = %s")
            query_params.append(workflow_id)
        
        # WHERE 절 구성
        where_clause = ""
        if where_conditions:
            where_clause = "WHERE " + " AND ".join(where_conditions)
        
        query = f"""
        SELECT * FROM execution_meta 
        {where_clause}
        ORDER BY updated_at DESC, created_at DESC
        LIMIT %s
        """
        query_params.append(limit)
        
        # SQLite인 경우 파라미터 플레이스홀더 변경
        if db_manager.config_db_manager.db_type == "sqlite":
            query = query.replace("%s", "?")
        
        # 쿼리 실행
        result = db_manager.config_db_manager.execute_query(query, tuple(query_params))
        
        if not result:
            return JSONResponse(content={
                "execution_meta_list": [],
                "total_count": 0,
                "filters": {
                    "interaction_id": interaction_id,
                    "workflow_id": workflow_id,
                    "limit": limit
                },
                "message": "No execution meta data found"
            })
        
        execution_meta_list = []
        for row in result:
            meta_data = {
                "id": row.get('id'),
                "interaction_id": row['interaction_id'],
                "workflow_id": row['workflow_id'],
                "workflow_name": row['workflow_name'],
                "interaction_count": row['interaction_count'],
                "metadata": json.loads(row['metadata']) if row.get('metadata') else {},
                "created_at": row['created_at'].isoformat() if isinstance(row['created_at'], datetime) else row['created_at'],
                "updated_at": row['updated_at'].isoformat() if isinstance(row['updated_at'], datetime) else row['updated_at']
            }
            execution_meta_list.append(meta_data)
        
        response_data = {
            "execution_meta_list": execution_meta_list,
            "total_count": len(execution_meta_list),
            "filters": {
                "interaction_id": interaction_id,
                "workflow_id": workflow_id,
                "limit": limit
            },
            "message": f"Found {len(execution_meta_list)} execution meta records"
        }
        
        logger.info(f"Retrieved {len(execution_meta_list)} execution meta records with filters: interaction_id={interaction_id}, workflow_id={workflow_id}")
        return JSONResponse(content=response_data)
        
    except Exception as e:
        logger.error(f"Error listing execution meta: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list execution meta: {str(e)}")

@router.post("/new", response_model=Dict[str, Any])
async def execute_workflow_new(request: Request, request_body: WorkflowRequest):
    """
    워크플로우를 실행하고 ExecutionMeta 테이블에 메타데이터를 저장합니다.
    """

    try:
        downloads_path = os.path.join(os.getcwd(), "downloads")
        if not request_body.workflow_name.endswith('.json'):
            filename = f"{request_body.workflow_name}.json"
        else:
            filename = request_body.workflow_name
        file_path = os.path.join(downloads_path, filename)

        with open(file_path, 'r', encoding='utf-8') as f:
            workflow_data = json.load(f)
            
        if workflow_data.get('workflow_id') != request_body.workflow_id:
            raise ValueError(f"워크플로우 ID가 일치하지 않습니다: {workflow_data.get('workflow_id')} != {request_body.workflow_id}")

        if not workflow_data or 'nodes' not in workflow_data or 'edges' not in workflow_data:
            raise ValueError(f"워크플로우 데이터가 유효하지 않습니다: {file_path}")
        
        print("--- 새 워크플로우 실행 시작 ---")
        print(f"실행 워크플로우: {request_body.workflow_name} ({request_body.workflow_id})")
        print(f"상호작용 ID: {request_body.interaction_id}")
        print(f"입력 데이터: {request_body.input_data}")

        if request_body.input_data is not None:
            for node in workflow_data.get('nodes', []):
                if node.get('data', {}).get('functionId') == 'startnode':
                    parameters = node.get('data', {}).get('parameters', [])
                    if parameters and isinstance(parameters, list):
                        parameters[0]['value'] = request_body.input_data
                        break
        
        # 데이터베이스 매니저 가져오기
        db_manager = None
        if hasattr(request.app.state, 'app_db') and request.app.state.app_db:
            db_manager = request.app.state.app_db
        else:
            raise HTTPException(status_code=500, detail="Database connection not available")

        # ExecutionMeta 조회 또는 생성
        execution_meta = await _get_or_create_execution_meta(
            db_manager,
            request_body.interaction_id,
            request_body.workflow_id,
            request_body.workflow_name,
            first_msg=request_body.input_data if request_body.input_data else None
        )

        # 워크플로우 실행
        executor = WorkflowExecutor(workflow_data, db_manager, request_body.interaction_id)
        final_outputs = executor.execute_workflow()

        # ExecutionMeta 업데이트 (interaction_count 증가)
        await _update_execution_meta_count(db_manager, execution_meta)

        return {
            "status": "success", 
            "message": "워크플로우 실행 완료", 
            "outputs": final_outputs,
            "execution_meta": {
                "interaction_id": execution_meta.interaction_id,
                "interaction_count": execution_meta.interaction_count + 1,
                "workflow_id": execution_meta.workflow_id,
                "workflow_name": execution_meta.workflow_name
            }
        }

    except ValueError as e:
        logging.error(f"Workflow execution error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@router.get("/io_logs")
async def get_workflow_io_logs(request: Request, workflow_name: str, workflow_id: str, interaction_id: str = 'default'):
    """
    특정 워크플로우의 ExecutionIO 로그를 반환합니다.
    
    Args:
        workflow_name: 워크플로우 이름
        workflow_id: 워크플로우 ID
        interaction_id: 상호작용 ID (선택적, 제공되지 않으면 default만 반환)
        
    Returns:
        ExecutionIO 로그 리스트
    """
    try:
        # 데이터베이스 매니저 가져오기
        if not hasattr(request.app.state, 'app_db') or not request.app.state.app_db:
            raise HTTPException(status_code=500, detail="Database connection not available")
        
        db_manager = request.app.state.app_db
        
        # SQL 쿼리 작성 - interaction_id 조건 추가
        if interaction_id:
            query = """
            SELECT
                interaction_id,
                workflow_name,
                workflow_id,
                input_data,
                output_data,
                updated_at
            FROM execution_io 
            WHERE workflow_name = %s AND workflow_id = %s AND interaction_id = %s
            ORDER BY updated_at
            """
            query_params = (workflow_name, workflow_id, interaction_id)
        else:
            query = """
            SELECT
                interaction_id,
                workflow_name,
                workflow_id,
                input_data,
                output_data,
                updated_at
            FROM execution_io 
            WHERE workflow_name = %s AND workflow_id = %s
            ORDER BY updated_at
            """
            query_params = (workflow_name, workflow_id)
        
        # SQLite인 경우 파라미터 플레이스홀더 변경
        if db_manager.config_db_manager.db_type == "sqlite":
            query = query.replace("%s", "?")
        
        # 쿼리 실행
        result = db_manager.config_db_manager.execute_query(query, query_params)
        
        if not result:
            logger.info(f"No performance data found for workflow: {workflow_name} ({workflow_id})")
            return JSONResponse(content={
                "workflow_name": workflow_name,
                "workflow_id": workflow_id,
                "in_out_logs": [],
                "message": "No in_out_logs data found for this workflow"
            })
        
        performance_stats = []
        for idx, row in enumerate(result):
            log_entry = {
                "log_id": idx + 1,
                "interaction_id": row['interaction_id'],
                "workflow_name": row['workflow_name'],
                "workflow_id": row['workflow_id'],
                "input_data": json.loads(row['input_data']).get('result', None) if row['input_data'] else None,
                "output_data": json.loads(row['output_data']).get('result', None) if row['output_data'] else None,
                "updated_at": row['updated_at'].isoformat() if isinstance(row['updated_at'], datetime) else row['updated_at']
            }
            performance_stats.append(log_entry)
        
        response_data = {
            "workflow_name": workflow_name,
            "workflow_id": workflow_id,
            "in_out_logs": performance_stats,
            "message": "In/Out logs retrieved successfully"
        }
        
        logger.info(f"Performance stats retrieved for workflow: {workflow_name} ({workflow_id})")
        return JSONResponse(content=response_data)
        
    except Exception as e:
        logger.error(f"Error retrieving workflow performance: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve performance data: {str(e)}")
    

@router.delete("/io_logs")
async def delete_workflow_io_logs(request: Request, workflow_name: str, workflow_id: str, interaction_id: str = "default"):
    """
    특정 워크플로우의 ExecutionIO 로그를 삭제합니다.
    
    Args:
        workflow_name: 워크플로우 이름
        workflow_id: 워크플로우 ID
        interaction_id: 상호작용 ID (기본값: "default")
        
    Returns:
        삭제된 로그 개수와 성공 메시지
    """
    try:
        # 데이터베이스 매니저 가져오기
        if not hasattr(request.app.state, 'app_db') or not request.app.state.app_db:
            raise HTTPException(status_code=500, detail="Database connection not available")
        
        db_manager = request.app.state.app_db
        
        # 먼저 삭제할 로그 개수 확인
        count_query = """
        SELECT COUNT(*) as count
        FROM execution_io 
        WHERE workflow_name = %s AND workflow_id = %s AND interaction_id = %s
        """
        
        # SQLite인 경우 파라미터 플레이스홀더 변경
        if db_manager.config_db_manager.db_type == "sqlite":
            count_query = count_query.replace("%s", "?")
        
        # 삭제할 로그 개수 조회
        count_result = db_manager.config_db_manager.execute_query(
            count_query, 
            (workflow_name, workflow_id, interaction_id)
        )
        
        delete_count = count_result[0]['count'] if count_result else 0
        
        if delete_count == 0:
            logger.info(f"No logs found to delete for workflow: {workflow_name} ({workflow_id}), interaction_id: {interaction_id}")
            return JSONResponse(content={
                "workflow_name": workflow_name,
                "workflow_id": workflow_id,
                "interaction_id": interaction_id,
                "deleted_count": 0,
                "message": "No logs found to delete"
            })
        
        # 삭제 쿼리 실행
        delete_query = """
        DELETE FROM execution_io 
        WHERE workflow_name = %s AND workflow_id = %s AND interaction_id = %s
        """
        
        # SQLite인 경우 파라미터 플레이스홀더 변경
        if db_manager.config_db_manager.db_type == "sqlite":
            delete_query = delete_query.replace("%s", "?")
        
        # 삭제 실행
        db_manager.config_db_manager.execute_query(
            delete_query, 
            (workflow_name, workflow_id, interaction_id)
        )
        
        response_data = {
            "workflow_name": workflow_name,
            "workflow_id": workflow_id,
            "interaction_id": interaction_id,
            "deleted_count": delete_count,
            "message": f"Successfully deleted {delete_count} execution logs"
        }
        
        logger.info(f"Deleted {delete_count} execution logs for workflow: {workflow_name} ({workflow_id}), interaction_id: {interaction_id}")
        return JSONResponse(content=response_data)
        
    except Exception as e:
        logger.error(f"Error deleting execution logs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete execution logs: {str(e)}")

@router.get("/execution_meta")
async def get_execution_meta(request: Request, interaction_id: str = "default", workflow_id: str = None):
    """
    ExecutionMeta 데이터를 조회합니다.
    
    Args:
        interaction_id: 상호작용 ID (기본값: "default")
        workflow_id: 워크플로우 ID (선택적, 제공되지 않으면 interaction_id의 모든 데이터 반환)
        
    Returns:
        ExecutionMeta 데이터 리스트
    """
    try:
        # 데이터베이스 매니저 가져오기
        if not hasattr(request.app.state, 'app_db') or not request.app.state.app_db:
            raise HTTPException(status_code=500, detail="Database connection not available")
        
        db_manager = request.app.state.app_db
        
        # SQL 쿼리 작성
        if workflow_id:
            query = """
            SELECT * FROM execution_meta 
            WHERE interaction_id = %s AND workflow_id = %s
            ORDER BY updated_at DESC
            """
            query_params = (interaction_id, workflow_id)
        else:
            query = """
            SELECT * FROM execution_meta 
            WHERE interaction_id = %s
            ORDER BY updated_at DESC
            """
            query_params = (interaction_id,)
        
        # SQLite인 경우 파라미터 플레이스홀더 변경
        if db_manager.config_db_manager.db_type == "sqlite":
            query = query.replace("%s", "?")
        
        # 쿼리 실행
        result = db_manager.config_db_manager.execute_query(query, query_params)
        
        if not result:
            return JSONResponse(content={
                "interaction_id": interaction_id,
                "workflow_id": workflow_id,
                "execution_meta": [],
                "message": "No execution meta data found"
            })
        
        execution_meta_list = []
        for row in result:
            meta_data = {
                "id": row.get('id'),
                "interaction_id": row['interaction_id'],
                "workflow_id": row['workflow_id'],
                "workflow_name": row['workflow_name'],
                "interaction_count": row['interaction_count'],
                "metadata": json.loads(row['metadata']) if row.get('metadata') else {},
                "created_at": row['created_at'].isoformat() if isinstance(row['created_at'], datetime) else row['created_at'],
                "updated_at": row['updated_at'].isoformat() if isinstance(row['updated_at'], datetime) else row['updated_at']
            }
            execution_meta_list.append(meta_data)
        
        response_data = {
            "interaction_id": interaction_id,
            "workflow_id": workflow_id,
            "execution_meta": execution_meta_list,
            "message": f"Found {len(execution_meta_list)} execution meta records"
        }
        
        logger.info(f"Retrieved {len(execution_meta_list)} execution meta records for interaction_id: {interaction_id}")
        return JSONResponse(content=response_data)
        
    except Exception as e:
        logger.error(f"Error retrieving execution meta: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve execution meta: {str(e)}")


@router.delete("/execution_meta")
async def delete_execution_meta(request: Request, interaction_id: str = "default", workflow_id: str = None):
    """
    ExecutionMeta 데이터를 삭제합니다.
    
    Args:
        interaction_id: 상호작용 ID (기본값: "default")
        workflow_id: 워크플로우 ID (선택적, 제공되지 않으면 interaction_id의 모든 데이터 삭제)
        
    Returns:
        삭제된 ExecutionMeta 개수와 성공 메시지
    """
    try:
        # 데이터베이스 매니저 가져오기
        if not hasattr(request.app.state, 'app_db') or not request.app.state.app_db:
            raise HTTPException(status_code=500, detail="Database connection not available")
        
        db_manager = request.app.state.app_db
        
        # 먼저 삭제할 데이터 개수 확인
        if workflow_id:
            count_query = """
            SELECT COUNT(*) as count FROM execution_meta 
            WHERE interaction_id = %s AND workflow_id = %s
            """
            query_params = (interaction_id, workflow_id)
        else:
            count_query = """
            SELECT COUNT(*) as count FROM execution_meta 
            WHERE interaction_id = %s
            """
            query_params = (interaction_id,)
        
        # SQLite인 경우 파라미터 플레이스홀더 변경
        if db_manager.config_db_manager.db_type == "sqlite":
            count_query = count_query.replace("%s", "?")
        
        # 삭제할 데이터 개수 조회
        count_result = db_manager.config_db_manager.execute_query(count_query, query_params)
        delete_count = count_result[0]['count'] if count_result else 0
        
        if delete_count == 0:
            logger.info(f"No execution meta found to delete for interaction_id: {interaction_id}, workflow_id: {workflow_id}")
            return JSONResponse(content={
                "interaction_id": interaction_id,
                "workflow_id": workflow_id,
                "deleted_count": 0,
                "message": "No execution meta found to delete"
            })
        
        # 삭제 쿼리 실행
        if workflow_id:
            delete_query = """
            DELETE FROM execution_meta 
            WHERE interaction_id = %s AND workflow_id = %s
            """
        else:
            delete_query = """
            DELETE FROM execution_meta 
            WHERE interaction_id = %s
            """
        
        # SQLite인 경우 파라미터 플레이스홀더 변경
        if db_manager.config_db_manager.db_type == "sqlite":
            delete_query = delete_query.replace("%s", "?")
        
        # 삭제 실행
        db_manager.config_db_manager.execute_query(delete_query, query_params)
        
        response_data = {
            "interaction_id": interaction_id,
            "workflow_id": workflow_id,
            "deleted_count": delete_count,
            "message": f"Successfully deleted {delete_count} execution meta records"
        }
        
        logger.info(f"Deleted {delete_count} execution meta records for interaction_id: {interaction_id}, workflow_id: {workflow_id}")
        return JSONResponse(content=response_data)
        
    except Exception as e:
        logger.error(f"Error deleting execution meta: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete execution meta: {str(e)}")

async def _get_or_create_execution_meta(db_manager, interaction_id: str, workflow_id: str, workflow_name: str, first_msg: str = None) -> ExecutionMeta:
    """ExecutionMeta를 조회하거나 새로 생성합니다."""
    try:
        # 기존 ExecutionMeta 조회
        query = """
        SELECT * FROM execution_meta 
        WHERE interaction_id = %s AND workflow_id = %s
        """
        
        # SQLite인 경우 파라미터 플레이스홀더 변경
        if db_manager.config_db_manager.db_type == "sqlite":
            query = query.replace("%s", "?")
        
        result = db_manager.config_db_manager.execute_query(query, (interaction_id, workflow_id))
        
        if result and len(result) > 0:
            # 기존 데이터가 있으면 반환
            row = result[0]
            execution_meta = ExecutionMeta(
                id=row.get('id'),
                interaction_id=row['interaction_id'],
                workflow_id=row['workflow_id'],
                workflow_name=row['workflow_name'],
                interaction_count=row['interaction_count'],
                metadata=json.loads(row['metadata']) if row.get('metadata') else {}
            )
            logger.info(f"Found existing ExecutionMeta for interaction_id: {interaction_id}, workflow_id: {workflow_id}")
            return execution_meta
        else:
            # 새로 생성
            execution_meta = ExecutionMeta(
                interaction_id=interaction_id,
                workflow_id=workflow_id,
                workflow_name=workflow_name,
                interaction_count=0,
                metadata={
                    "placeholder": first_msg if first_msg else "No initial message provided"
                }
            )
            
            # DB에 저장
            insert_query = """
            INSERT INTO execution_meta (interaction_id, workflow_id, workflow_name, interaction_count, metadata, created_at)
            VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
            """
            
            if db_manager.config_db_manager.db_type == "sqlite":
                insert_query = insert_query.replace("%s", "?")
            
            metadata_json = json.dumps(execution_meta.metadata, ensure_ascii=False)
            db_manager.config_db_manager.execute_query(
                insert_query, 
                (interaction_id, workflow_id, workflow_name, 0, metadata_json)
            )
            
            logger.info(f"Created new ExecutionMeta for interaction_id: {interaction_id}, workflow_id: {workflow_id}")
            return execution_meta
            
    except Exception as e:
        logger.error(f"Error handling ExecutionMeta: {str(e)}")
        # 실패해도 기본값으로 계속 진행
        return ExecutionMeta(
            interaction_id=interaction_id,
            workflow_id=workflow_id,
            workflow_name=workflow_name,
            interaction_count=0,
            metadata={}
        )


async def _update_execution_meta_count(db_manager, execution_meta: ExecutionMeta):
    """ExecutionMeta의 interaction_count를 1 증가시킵니다."""
    try:
        update_query = """
        UPDATE execution_meta 
        SET interaction_count = interaction_count + 1, updated_at = CURRENT_TIMESTAMP
        WHERE interaction_id = %s AND workflow_id = %s
        """
        
        if db_manager.config_db_manager.db_type == "sqlite":
            update_query = update_query.replace("%s", "?")
        
        db_manager.config_db_manager.execute_query(
            update_query, 
            (execution_meta.interaction_id, execution_meta.workflow_id)
        )
        
        logger.info(f"Updated interaction_count for interaction_id: {execution_meta.interaction_id}, workflow_id: {execution_meta.workflow_id}")
        
    except Exception as e:
        logger.error(f"Error updating ExecutionMeta count: {str(e)}")
        # 카운트 업데이트 실패해도 전체 실행은 계속 진행


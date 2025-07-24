from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import json
import logging
from datetime import datetime
from editor.workflow_executor import WorkflowExecutor
from service.database.execution_meta_service import get_or_create_execution_meta, update_execution_meta_count
from service.general_function import create_conversation_function

from service.database.models.executor import ExecutionMeta, ExecutionIO
from service.database.models.workflow import WorkflowMeta
from service.database.models.performance import NodePerformance

logger = logging.getLogger("workflow-controller")
router = APIRouter(prefix="/api/workflow", tags=["workflow"])

class WorkflowRequest(BaseModel):
    workflow_name: str
    workflow_id: str
    input_data: str = ""
    interaction_id: str = "default"
    selected_collection: Optional[str] = None

class WorkflowData(BaseModel):
    workflow_name: str
    workflow_id: str
    view: Dict[str, Any]
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    interaction_id: str = "default"

class SaveWorkflowRequest(BaseModel):
    workflow_name: str
    content: WorkflowData

class ConversationRequest(BaseModel):
    """통합 대화/워크플로우 실행 요청 모델"""
    user_input: str
    interaction_id: str
    execution_type: str = "default_mode"  # "default_mode" 또는 "workflow"
    workflow_id: Optional[str] = None
    workflow_name: Optional[str] = None
    selected_collection: Optional[str] = None

def get_db_manager(request: Request):
    """데이터베이스 매니저 의존성 주입"""
    if hasattr(request.app.state, 'app_db') and request.app.state.app_db:
        return request.app.state.app_db
    else:
        raise HTTPException(status_code=500, detail="Database connection not available")

def get_rag_service(request: Request):
    """RAG 서비스 의존성 주입"""
    if hasattr(request.app.state, 'rag_service') and request.app.state.rag_service:
        return request.app.state.rag_service
    else:
        raise HTTPException(status_code=500, detail="RAG service not available")

@router.get("/list")
async def list_workflows(request: Request):
    """
    downloads 폴더에 있는 모든 workflow 파일들의 이름을 반환합니다.
    """
    try:
        user_id = request.headers.get("X-User-ID")
        token = request.headers.get("Authorization")


        downloads_path = os.path.join(os.getcwd(), "downloads")
        download_path_id = os.path.join(downloads_path, user_id)

        # downloads 폴더가 존재하지 않으면 생성
        if not os.path.exists(download_path_id):
            os.makedirs(download_path_id)
            return JSONResponse(content={"workflows": []})

        # .json 확장자를 가진 파일들만 필터링
        workflow_files = []
        for file in os.listdir(download_path_id):
            if file.endswith('.json'):
                workflow_files.append(file)

        logger.info(f"Found {len(workflow_files)} workflow files")
        return JSONResponse(content={"workflows": workflow_files})

    except Exception as e:
        logger.error(f"Error listing workflows: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list workflows: {str(e)}")

@router.post("/save")
async def save_workflow(request: Request, workflow_request: SaveWorkflowRequest):
    """
    Frontend에서 받은 workflow 정보를 파일로 저장합니다.
    파일명: {workflow_name}.json
    """
    try:
        user_id = request.headers.get("X-User-ID")
        token = request.headers.get("Authorization")

        downloads_path = os.path.join(os.getcwd(), "downloads")
        download_path_id = os.path.join(downloads_path, user_id)
        workflow_data = workflow_request.content.dict()
        if not os.path.exists(download_path_id):
            os.makedirs(download_path_id)

        if not workflow_request.workflow_name.endswith('.json'):
            filename = f"{workflow_request.workflow_name}.json"
        else:
            filename = workflow_request.workflow_name
        file_path = os.path.join(download_path_id, filename)

        app_db = get_db_manager(request)

        # nodes 수 계산
        nodes = workflow_data.get('nodes', [])
        node_count = len(nodes) if isinstance(nodes, list) else 0
        has_startnode = any(
            node.get('data', {}).get('functionId') == 'startnode' for node in nodes
        )
        has_endnode = any(
            node.get('data', {}).get('functionId') == 'endnode' for node in nodes
        )

        # edges 수 계산
        edges = workflow_data.get('edges', [])
        edge_count = len(edges) if isinstance(edges, list) else 0

        workflow_meta = WorkflowMeta(
            user_id=user_id,
            workflow_id=workflow_request.content.workflow_id,
            workflow_name=workflow_request.workflow_name,
            node_count=node_count,
            edge_count=edge_count,
            has_startnode=has_startnode,
            has_endnode=has_endnode,
            is_completed=(has_startnode and has_endnode),
        )

        existing_data = app_db.find_by_condition(
            WorkflowMeta,
            {
                "user_id": user_id,
                "workflow_id": workflow_request.content.workflow_id,
                "workflow_name": workflow_request.workflow_name,
                "node_count": node_count,
                "edge_count": edge_count,
                "has_startnode": has_startnode,
                "has_endnode": has_endnode,
                "is_completed": (has_startnode and has_endnode),
            },
            limit=1
        )
        if existing_data:
            existing_data_id = existing_data[0].id
            workflow_meta.id = existing_data_id
            insert_result = app_db.update(workflow_meta)
        else:
            insert_result = app_db.insert(workflow_meta)


        if insert_result and insert_result.get("result") == "success":
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(workflow_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Workflow metadata saved successfully: {workflow_request.workflow_name}")
        else:
            logger.error(f"Failed to save workflow metadata: {insert_result.get('error', 'Unknown error')}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to save workflow metadata: {insert_result.get('error', 'Unknown error')}"
            )

        logger.info(f"Workflow saved successfully: {filename}")
        return JSONResponse(content={
            "success": True,
            "message": f"Workflow '{workflow_request.workflow_name}' saved successfully",
            "filename": filename
        })

    except Exception as e:
        logger.error(f"Error saving workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save workflow: {str(e)}")

@router.get("/load/{workflow_id}")
async def load_workflow(request: Request, workflow_id: str):
    """
    특정 workflow를 로드합니다.
    """
    try:
        user_id = request.headers.get("X-User-ID")
        token = request.headers.get("Authorization")

        downloads_path = os.path.join(os.getcwd(), "downloads")
        download_path_id = os.path.join(downloads_path, user_id)

        filename = f"{workflow_id}.json"
        file_path = os.path.join(download_path_id, filename)

        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"Workflow '{workflow_id}' not found")

        with open(file_path, 'r', encoding='utf-8') as f:
            workflow_data = json.load(f)

        logger.info(f"Workflow loaded successfully: {filename}")
        return JSONResponse(content=workflow_data)

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Workflow '{workflow_id}' not found")
    except Exception as e:
        logger.error(f"Error loading workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load workflow: {str(e)}")

@router.delete("/delete/{workflow_name}")
async def delete_workflow(request: Request, workflow_name: str):
    """
    특정 workflow를 삭제합니다.
    """
    try:
        user_id = request.headers.get("X-User-ID")
        token = request.headers.get("Authorization")
        app_db = get_db_manager(request)

        existing_data = app_db.find_by_condition(
            WorkflowMeta,
            {
                "user_id": user_id,
                "workflow_name": workflow_name,
            },
            limit=1
        )

        downloads_path = os.path.join(os.getcwd(), "downloads")
        download_path_id = os.path.join(downloads_path, user_id)
        filename = f"{workflow_name}.json"
        file_path = os.path.join(download_path_id, filename)

        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"Workflow '{workflow_name}' not found")

        app_db.delete(WorkflowMeta, existing_data[0].id if existing_data else None)
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

@router.get("/list/detail")
async def list_workflows_detail(request: Request):
    """
    downloads 폴더에 있는 모든 workflow 파일들의 상세 정보를 반환합니다.
    각 워크플로우에 대해 파일명, workflow_id, 노드 수, 마지막 수정일자를 포함합니다.
    """
    try:
        user_id = request.headers.get("X-User-ID")
        token = request.headers.get("Authorization")
        app_db = get_db_manager(request)

        existing_data = app_db.find_by_condition(
            WorkflowMeta,
            {
                "user_id": user_id,
            },
            limit=10000
        )

        response_data = []
        for data in existing_data:
            data_dict = data.to_dict()
            response_data.append(data_dict)

        logger.info(f"Found {len(existing_data)} workflow files with detailed information")

        return JSONResponse(content={"workflows": response_data})

    except Exception as e:
        logger.error(f"Error listing workflow details: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list workflow details: {str(e)}")

@router.get("/performance")
async def get_workflow_performance(request: Request, workflow_name: str, workflow_id: str):
    """
    특정 워크플로우의 성능 통계를 반환합니다.
    node_id와 node_name별로 평균 성능 지표를 계산합니다.

    Args:
        workflow_name: 워크플로우 이름
        workflow_id: 워크플로우 ID

    Returns:
        노드별 성능 통계와 전체 워크플로우 통계
    """
    try:
        user_id = request.headers.get("X-User-ID")
        token = request.headers.get("Authorization")
        app_db = get_db_manager(request)

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
            logger.info(f"No performance data found for workflow: {workflow_name} ({workflow_id})")
            return JSONResponse(content={
                "workflow_name": workflow_name,
                "workflow_id": workflow_id,
                "performance_stats": [],
                "message": "No performance data found for this workflow"
            })

        # Decimal 타입을 float로 변환하는 헬퍼 함수
        def safe_round_float(value, decimal_places=4):
            if value is None:
                return None
            try:
                # Decimal, float, int, str 모든 타입을 float로 변환
                if hasattr(value, '__float__'):  # Decimal 포함
                    return round(float(value), decimal_places)
                elif isinstance(value, (int, float)):
                    return round(float(value), decimal_places)
                elif isinstance(value, str):
                    return round(float(value), decimal_places)
                else:
                    return float(value) if value else 0.0
            except (ValueError, TypeError):
                return 0.0

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

        # GPU 통계 (GPU 데이터가 있는 경우만)
        gpu_stats = None
        total_gpu_executions = sum(stat['gpu_execution_count'] for stat in performance_stats)
        if total_gpu_executions > 0:
            gpu_processing_time_sum = sum(float(stat['avg_processing_time_ms']) * stat['gpu_execution_count'] for stat in performance_stats if stat['avg_gpu_usage_percent'] is not None)
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

        logger.info(f"Performance stats retrieved for workflow: {workflow_name} ({workflow_id})")
        return JSONResponse(content=response_data)

    except Exception as e:
        logger.error(f"Error retrieving workflow performance: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve performance data: {str(e)}")

@router.delete("/performance")
async def delete_workflow_performance(request: Request, workflow_name: str, workflow_id: str):
    """
    특정 워크플로우의 성능 데이터를 삭제합니다.

    Args:
        workflow_name: 워크플로우 이름
        workflow_id: 워크플로우 ID

    Returns:
        삭제된 성능 데이터 개수와 성공 메시지
    """
    try:
        user_id = request.headers.get("X-User-ID")
        token = request.headers.get("Authorization")
        app_db = get_db_manager(request)
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

        logger.info(f"Deleted {delete_count} performance records for workflow: {workflow_name} ({workflow_id})")
        return JSONResponse(content=response_data)

    except Exception as e:
        logger.error(f"Error deleting performance data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete performance data: {str(e)}")

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
        user_id = request.headers.get("X-User-ID")
        token = request.headers.get("Authorization")
        app_db = get_db_manager(request)
        result = app_db.find_by_condition(
            ExecutionIO,
            {
                "user_id": user_id,
                "workflow_name": workflow_name,
                "workflow_id": workflow_id,
                "interaction_id": interaction_id
            },
            limit=1000000,  # 필요에 따라 조정 가능
            orderby="updated_at",
            orderby_asc=True,
            return_list=True
        )

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
        user_id = request.headers.get("X-User-ID")
        token = request.headers.get("Authorization")
        app_db = get_db_manager(request)

        existing_data = app_db.find_by_condition(
            ExecutionIO,
            {
                "user_id": user_id,
                "workflow_name": workflow_name,
                "workflow_id": workflow_id,
                "interaction_id": interaction_id
            },
            limit=1000000
        )

        delete_count = len(existing_data) if existing_data else 0

        if delete_count == 0:
            logger.info(f"No logs found to delete for workflow: {workflow_name} ({workflow_id}), interaction_id: {interaction_id}")
            return JSONResponse(content={
                "workflow_name": workflow_name,
                "workflow_id": workflow_id,
                "interaction_id": interaction_id,
                "deleted_count": 0,
                "message": "No logs found to delete"
            })

        app_db.delete_by_condition(
            ExecutionIO,
            {
                "user_id": user_id,
                "workflow_name": workflow_name,
                "workflow_id": workflow_id,
                "interaction_id": interaction_id
            }
        )
        app_db.delete_by_condition(
            ExecutionMeta,
            {
                "user_id": user_id,
                "workflow_name": workflow_name,
                "workflow_id": workflow_id,
                "interaction_id": interaction_id
            }
        )

        logger.info(f"Successfully deleted {delete_count} logs for workflow: {workflow_name} ({workflow_id}), interaction_id: {interaction_id}")

        return JSONResponse(content={
            "workflow_name": workflow_name,
            "workflow_id": workflow_id,
            "interaction_id": interaction_id,
            "deleted_count": delete_count,
            "message": f"Successfully deleted {delete_count} execution logs"
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting workflow logs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete workflow logs: {str(e)}")

@router.post("/execute", response_model=Dict[str, Any])
async def execute_workflow(request: Request, workflow: WorkflowData):
    """
    주어진 노드와 엣지 정보로 워크플로우를 실행합니다.
    """

    # print("DEBUG: 워크플로우 실행 요청\n", workflow)

    try:
        user_id = request.headers.get("X-User-ID")
        token = request.headers.get("Authorization")
        workflow_data = workflow.dict()
        app_db = get_db_manager(request)

        executor = WorkflowExecutor(workflow_data, app_db, workflow.interaction_id, user_id)
        final_outputs = executor.execute_workflow()

        return {"status": "success", "message": "워크플로우 실행 완료", "outputs": final_outputs}

    except ValueError as e:
        logging.error(f"Workflow execution error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@router.post("/execute/based_id", response_model=Dict[str, Any])
async def execute_workflow_with_id(request: Request, request_body: WorkflowRequest):
    """
    주어진 노드와 엣지 정보로 워크플로우를 실행합니다.
    """
    try:
        user_id = request.headers.get("X-User-ID")
        token = request.headers.get("Authorization")

        ## 일반채팅인 경우 미리 정의된 워크플로우를 이용하여 일반 채팅에 사용.
        if request_body.workflow_name == 'default_mode':
            default_mode_workflow_folder = os.path.join(os.getcwd(), "constants")
            file_path = os.path.join(default_mode_workflow_folder, "base_chat_workflow.json")
            with open(file_path, 'r', encoding='utf-8') as f:
                workflow_data = json.load(f)
            workflow_data = await _default_workflow_parameter_helper(request, request_body, workflow_data)

        ## 워크플로우 실행인 경우, 해당하는 워크플로우 파일을 찾아서 사용.
        else:
            downloads_path = os.path.join(os.getcwd(), "downloads")
            download_path_id = os.path.join(downloads_path, user_id)

            if not request_body.workflow_name.endswith('.json'):
                filename = f"{request_body.workflow_name}.json"
            else:
                filename = request_body.workflow_name
            file_path = os.path.join(download_path_id, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                workflow_data = json.load(f)
            workflow_data = await _workflow_parameter_helper(request_body, workflow_data)

        ## ========== 워크플로우 데이터 검증 ==========
        if workflow_data.get('workflow_id') != request_body.workflow_id:
            raise ValueError(f"워크플로우 ID가 일치하지 않습니다: {workflow_data.get('workflow_id')} != {request_body.workflow_id}")

        if not workflow_data or 'nodes' not in workflow_data or 'edges' not in workflow_data:
            raise ValueError(f"워크플로우 데이터가 유효하지 않습니다: {file_path}")

        print("--- 워크플로우 실행 시작 ---")
        print(f"실행 워크플로우: {request_body.workflow_name} ({request_body.workflow_id})")
        print(f"입력 데이터: {request_body.input_data}")

        ## 모든 워크플로우는 startnode가 있어야 하며, 입력 데이터는 startnode의 첫 번째 파라미터로 설정되어야 함.
        ## 사용자의 인풋은 여기에 입력되고, 워크플로우가 실행됨.
        if request_body.input_data is not None:
            for node in workflow_data.get('nodes', []):
                if node.get('data', {}).get('functionId') == 'startnode':
                    parameters = node.get('data', {}).get('parameters', [])
                    if parameters and isinstance(parameters, list):
                        parameters[0]['value'] = request_body.input_data
                        break

        ## app에 저장된 db_manager를 가져옴. 이걸 통해 DB에 접근할 수 있음.
        ## DB에 접근하여 execution 데이터를 활용하여, 기록된 대화를 가져올지 말지 결정.
        app_db = get_db_manager(request)

        ## 일반적인 실행(execution)이 아닌 경우, 즉 대화형 실행(conversation execution)인 경우
        ## interaction_id가 "default"가 아닌 경우, 대화형 실행을 위한 메타데이터를 가져오거나 생성
        ## interaction_id가 "default"인 경우, execution_meta는 None으로 설정
        execution_meta = None
        if request_body.interaction_id != "default" and app_db:
            execution_meta = await get_or_create_execution_meta(
                app_db,
                user_id,
                request_body.interaction_id,
                request_body.workflow_id,
                request_body.workflow_name,
                request_body.input_data
            )

        ## 워크플로우를 실질적으로 실행 (가장중요)
        ## 워크플로우 실행 관련 로직은 WorkflowExecutor 클래스에 정의되어 있음.
        ## WorkflowExecutor 클래스는 워크플로우의 노드와 엣지를 기반으로 워크플로우를 실행하는 역할을 함.
        ## 워크플로우 실행 시, interaction_id를 전달하여 대화형 실행을 지원함. (이렇게 되는 경우, interaction_id는 대화형 실행의 ID로 사용되어 DB에 저장됨)
        ## 워크플로우 실행 결과는 final_outputs에 저장됨.
        executor = WorkflowExecutor(workflow_data, app_db, request_body.interaction_id, user_id)
        final_outputs = executor.execute_workflow()

        ## 대화형 실행인 경우 execution_meta의 값을 가지고, 이 경우에는 대화 count를 증가.
        if execution_meta:
            await update_execution_meta_count(app_db, execution_meta)

        response_data = {"status": "success", "message": "워크플로우 실행 완료", "outputs": final_outputs}

        if execution_meta:
            response_data["execution_meta"] = {
                "interaction_id": execution_meta.interaction_id,
                "interaction_count": execution_meta.interaction_count + 1,
                "workflow_id": execution_meta.workflow_id,
                "workflow_name": execution_meta.workflow_name
            }

        return response_data

    except ValueError as e:
        logging.error(f"Workflow execution error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

# Helper Functions

async def _workflow_parameter_helper(request_body, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Updates the workflow data by setting the interaction ID in the parameters of nodes
    with a function ID of 'memory', if applicable.

    Args:
        request_body: An object containing the interaction ID to be applied.
        workflow_data: A dictionary representing the workflow's nodes and their parameters.

    Returns:
        The updated workflow data with the interaction ID applied where necessary.
    """
    if (request_body.interaction_id) and (request_body.interaction_id != "default"):
        for node in workflow_data.get('nodes', []):
            if node.get('data', {}).get('functionId') == 'memory':
                parameters = node.get('data', {}).get('parameters', [])
                for parameter in parameters:
                    if parameter.get('id') == 'interaction_id':
                        parameter['value'] = request_body.interaction_id

    return workflow_data

async def _default_workflow_parameter_helper(request, request_body, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Updates the workflow data with default parameters based on the request body.

    This function modifies the `workflow_data` dictionary by setting specific parameter values
    for nodes in the workflow. It updates the `collection_name` parameter for nodes with the
    `document_loaders` function ID and the `interaction_id` parameter for nodes with the
    `memory` function ID, based on the values provided in the `request_body`.

    Parameters:
        request_body: An object containing the request data. It should have attributes
            `selected_collection` (optional) and `interaction_id` (optional).
        workflow_data: A dictionary representing the workflow structure. It contains a list
            of nodes, each of which may have a `data` dictionary with `functionId` and `parameters`.

    Returns:
        A dictionary representing the updated workflow data with modified parameters.
    """
    config_composer = request.app.state.config_composer
    if not config_composer:
        raise HTTPException(status_code=500, detail="Config composer not available")

    llm_provider = config_composer.get_config_by_name("DEFAULT_LLM_PROVIDER").value

    if llm_provider == "openai":
        model = config_composer.get_config_by_name("OPENAI_MODEL_DEFAULT").value
        url = config_composer.get_config_by_name("OPENAI_API_BASE_URL").value
    elif llm_provider == "vllm":
        model = config_composer.get_config_by_name("VLLM_MODEL_NAME").value
        url = config_composer.get_config_by_name("VLLM_API_BASE_URL").value
    else:
        raise HTTPException(status_code=500, detail="Unsupported LLM provider")

    for node in workflow_data.get('nodes', []):
        if node.get('data', {}).get('functionId') == 'agents':
            parameters = node.get('data', {}).get('parameters', [])
            for parameter in parameters:
                if parameter.get('id') == 'model':
                    parameter['value'] = model
                if parameter.get('id') == 'base_url':
                    parameter['value'] = url

    if request_body.selected_collection:
        for node in workflow_data.get('nodes', []):
            if node.get('data', {}).get('functionId') == 'document_loaders':
                parameters = node.get('data', {}).get('parameters', [])
                for parameter in parameters:
                    if parameter.get('id') == 'collection_name':
                        parameter['value'] = request_body.selected_collection

    if (request_body.interaction_id) and (request_body.interaction_id != "default"):
        for node in workflow_data.get('nodes', []):
            if node.get('data', {}).get('functionId') == 'memory':
                parameters = node.get('data', {}).get('parameters', [])
                for parameter in parameters:
                    if parameter.get('id') == 'interaction_id':
                        parameter['value'] = request_body.interaction_id

    return workflow_data

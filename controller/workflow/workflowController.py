import asyncio
from fastapi import APIRouter, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import hashlib
import os
import copy
import json
import logging
import re
from datetime import datetime
from editor.async_workflow_executor import AsyncWorkflowExecutor, execution_manager
from service.database.execution_meta_service import get_or_create_execution_meta, update_execution_meta_count
from controller.controller_helper import extract_user_id_from_request
from controller.workflow.helper import _workflow_parameter_helper, _default_workflow_parameter_helper
from controller.workflow.model import WorkflowRequest, WorkflowData, SaveWorkflowRequest, ConversationRequest

from service.database.models.user import User
from service.database.models.executor import ExecutionMeta, ExecutionIO
from service.database.models.workflow import WorkflowMeta
from service.database.models.performance import NodePerformance

import uuid
import time

logger = logging.getLogger("workflow-controller")
router = APIRouter(prefix="/api/workflow", tags=["workflow"])

def extract_collection_name(collection_full_name: str) -> str:
    """
    컬렉션 이름에서 UUID 부분을 제거하고 실제 이름만 추출합니다.

    예: '장하렴연구_3a6a552d-d277-490d-9f3c-cead80d651f7' -> '장하렴연구'

    Args:
        collection_full_name: UUID가 포함된 전체 컬렉션 이름

    Returns:
        UUID 부분이 제거된 깨끗한 컬렉션 이름
    """
    # UUID 패턴: 8-4-4-4-12 형태의 16진수 문자열
    uuid_pattern = r'_[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'

    # UUID 부분을 제거하고 앞의 이름만 반환
    clean_name = re.sub(uuid_pattern, '', collection_full_name, flags=re.IGNORECASE)

    return clean_name

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
        user_id = extract_user_id_from_request(request)

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
        user_id = extract_user_id_from_request(request)

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
                "workflow_name": workflow_request.workflow_name,
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
        user_id = extract_user_id_from_request(request)

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
        user_id = extract_user_id_from_request(request)
        app_db = get_db_manager(request)

        existing_data = app_db.find_by_condition(
            WorkflowMeta,
            {
                "user_id": user_id,
                "workflow_name": workflow_name,
            },
            limit=1
        )

        app_db.delete(WorkflowMeta, existing_data[0].id if existing_data else None)

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

@router.get("/list/detail")
async def list_workflows_detail(request: Request):
    """
    downloads 폴더에 있는 모든 workflow 파일들의 상세 정보를 반환합니다.
    각 워크플로우에 대해 파일명, workflow_id, 노드 수, 마지막 수정일자를 포함합니다.
    """
    try:
        user_id = extract_user_id_from_request(request)
        app_db = get_db_manager(request)

        existing_data = app_db.find_by_condition(
            WorkflowMeta,
            {
                "user_id": user_id,
            },
            limit=10000,
            orderby="updated_at",
        )

        user_data = app_db.find_by_condition(User,{"id": user_id}, limit=1,
        )
        user_name = user_data[0].username if user_data else "Unknown User"

        response_data = []
        for data in existing_data:
            data_dict = data.to_dict()
            data_dict['user_name'] = user_name
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
        user_id = extract_user_id_from_request(request)
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
        user_id = extract_user_id_from_request(request)
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
        user_id = extract_user_id_from_request(request)
        app_db = get_db_manager(request)
        result = app_db.find_by_condition(
            ExecutionIO,
            {
                "user_id": user_id,
                "workflow_name": workflow_name,
                # "workflow_id": workflow_id, # workflow_id 로직 삭제
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
        user_id = extract_user_id_from_request(request)
        app_db = get_db_manager(request)

        existing_data = app_db.find_by_condition(
            ExecutionIO,
            {
                "user_id": user_id,
                "workflow_name": workflow_name,
                # "workflow_id": workflow_id, # workflow_id 로직 삭제
                "interaction_id": interaction_id
            },
            limit=1000000
        )

        delete_count = len(existing_data) if existing_data else 0

        if delete_count == 0:
            logger.info(f"No logs found to delete for workflow: {workflow_name} ({workflow_id}), interaction_id: {interaction_id}")
            return JSONResponse(content={
                "workflow_name": workflow_name,
                # "workflow_id": workflow_id,
                "interaction_id": interaction_id,
                "deleted_count": 0,
                "message": "No logs found to delete"
            })

        app_db.delete_by_condition(
            ExecutionIO,
            {
                "user_id": user_id,
                "workflow_name": workflow_name,
                # "workflow_id": workflow_id, # workflow_id 로직 삭제
                "interaction_id": interaction_id
            }
        )
        app_db.delete_by_condition(
            ExecutionMeta,
            {
                "user_id": user_id,
                "workflow_name": workflow_name,
                # "workflow_id": workflow_id, # workflow_id 로직 삭제
                "interaction_id": interaction_id
            }
        )

        logger.info(f"Successfully deleted {delete_count} logs for workflow: {workflow_name} ({workflow_id}), interaction_id: {interaction_id}")

        return JSONResponse(content={
            "workflow_name": workflow_name,
            # "workflow_id": workflow_id,
            "interaction_id": interaction_id,
            "deleted_count": delete_count,
            "message": f"Successfully deleted {delete_count} execution logs"
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting workflow logs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete workflow logs: {str(e)}")

# @router.post("/execute", response_model=Dict[str, Any])
# async def execute_workflow(request: Request, workflow: WorkflowData):
#     """
#     주어진 노드와 엣지 정보로 워크플로우를 비동기적으로 실행합니다.
#     """

#     # print("DEBUG: 워크플로우 실행 요청\n", workflow)

#     try:
#         user_id = extract_user_id_from_request(request)
#         workflow_data = workflow.dict()
#         app_db = get_db_manager(request)

#         # 비동기 실행기 생성
#         executor = execution_manager.create_executor(
#             workflow_data=workflow_data,
#             db_manager=app_db,
#             interaction_id=workflow.interaction_id,
#             user_id=user_id
#         )

#         # 백그라운드에서 비동기 실행
#         final_outputs = []
#         async for output in executor.execute_workflow_async():
#             final_outputs.append(output)

#         return {"status": "success", "message": "워크플로우 실행 완료", "outputs": final_outputs}

#     except ValueError as e:
#         logging.error(f"Workflow execution error: {e}")
#         raise HTTPException(status_code=400, detail=str(e))
#     except Exception as e:
#         logging.error(f"An unexpected error occurred: {e}")
#         raise HTTPException(status_code=500, detail="Internal Server Error")
#     finally:
#         # 완료된 실행들 정리
#         execution_manager.cleanup_completed_executions()

# @router.post("/execute/stream")
# async def execute_workflow_stream(request: Request, workflow: WorkflowData):
#     """
#     주어진 워크플로우를 비동기적으로 실행하고, 각 노드의 실행 결과를 SSE로 스트리밍합니다.
#     """

#     async def stream_generator(async_result_generator):
#         full_response_chunks = []
#         try:
#             async for chunk in async_result_generator:
#                 # 클라이언트에 보낼 데이터 형식 정의 (JSON)
#                 full_response_chunks.append(str(chunk))
#                 response_chunk = {"type": "data", "content": chunk}
#                 yield f"data: {json.dumps(response_chunk, ensure_ascii=False)}\n\n"
#                 await asyncio.sleep(0.01) # 짧은 딜레이로 이벤트 스트림 안정화

#             end_message = {"type": "end", "message": "Stream finished"}
#             yield f"data: {json.dumps(end_message)}\n\n"

#         except Exception as e:
#             logger.error(f"스트리밍 중 오류 발생: {e}", exc_info=True)
#             error_message = {"type": "error", "detail": f"스트리밍 중 오류가 발생했습니다: {str(e)}"}
#             yield f"data: {json.dumps(error_message)}\n\n"
#         finally:
#             # 완료된 실행들 정리
#             execution_manager.cleanup_completed_executions()

#     try:
#         user_id = extract_user_id_from_request(request)
#         workflow_data = workflow.dict()
#         app_db = get_db_manager(request)

#         # 비동기 실행기 생성
#         executor = execution_manager.create_executor(
#             workflow_data=workflow_data,
#             db_manager=app_db,
#             interaction_id=workflow.interaction_id,
#             user_id=user_id
#         )

#         # 비동기 제너레이터 시작 (스트리밍용)
#         result_generator = executor.execute_workflow_async_streaming()

#     except Exception as e:
#         # 스트림 시작 전 초기 설정에서 에러 발생 시
#         logging.error(f"Workflow pre-execution error: {e}")
#         raise HTTPException(status_code=400, detail=f"Error setting up workflow: {e}")

#     return StreamingResponse(stream_generator(result_generator), media_type="text/event-stream")

@router.post("/execute/based_id", response_model=Dict[str, Any])
async def execute_workflow_with_id(request: Request, request_body: WorkflowRequest):
    """
    주어진 노드와 엣지 정보로 워크플로우를 실행합니다.
    """
    try:
        user_id = extract_user_id_from_request(request)

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
        ## TODO 워크플로우 아이디 정합성 관련 로직 생각해볼 것
        # if workflow_data.get('workflow_id') != request_body.workflow_id:
        #     raise ValueError(f"워크플로우 ID가 일치하지 않습니다: {workflow_data.get('workflow_id')} != {request_body.workflow_id}")

        if not workflow_data or 'nodes' not in workflow_data or 'edges' not in workflow_data:
            raise ValueError(f"워크플로우 데이터가 유효하지 않습니다: {file_path}")

        print("--- 워크플로우 실행 시작 ---")
        print(f"실행 워크플로우: {request_body.workflow_name} ({request_body.workflow_id})")
        print(f"입력 데이터: {request_body.input_data}")

        ## 모든 워크플로우는 startnode가 있어야 하며, 입력 데이터는 startnode의 첫 번째 파라미터로 설정되어야 함.
        ## 사용자의 인풋은 여기에 입력되고, 워크플로우가 실행됨.
        if request_body.input_data is not None and request_body.input_data != "" and len(request_body.input_data) > 0:
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

        ## 워크플로우를 실질적으로 비동기 실행 (가장중요)
        ## 비동기 워크플로우 실행 관련 로직은 AsyncWorkflowExecutor 클래스에 정의되어 있음.
        ## AsyncWorkflowExecutor 클래스는 워크플로우의 노드와 엣지를 기반으로 워크플로우를 백그라운드에서 실행하는 역할을 함.
        ## 워크플로우 실행 시, interaction_id를 전달하여 대화형 실행을 지원함. (이렇게 되는 경우, interaction_id는 대화형 실행의 ID로 사용되어 DB에 저장됨)
        ## 워크플로우 실행 결과는 final_outputs에 저장됨.
        executor = execution_manager.create_executor(
            workflow_data=workflow_data,
            db_manager=app_db,
            interaction_id=request_body.interaction_id,
            user_id=user_id
        )

        # 비동기 실행 및 결과 수집
        final_outputs = []
        async for output in executor.execute_workflow_async():
            final_outputs.append(output)

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
    finally:
        # 완료된 실행들 정리
        execution_manager.cleanup_completed_executions()

@router.post("/execute/based_id/stream")
async def execute_workflow_with_id_stream(request: Request, request_body: WorkflowRequest):
    """
    주어진 ID를 기반으로 워크플로우를 스트리밍 방식으로 실행합니다.
    """
    user_id = extract_user_id_from_request(request)
    app_db = get_db_manager(request)

    async def stream_generator(async_result_generator, db_manager, user_id, workflow_req):
        """결과 제너레이터를 SSE 형식으로 변환하는 비동기 제너레이터"""
        full_response_chunks = []
        try:
            async for chunk in async_result_generator:
                # 클라이언트에 보낼 데이터 형식 정의 (JSON)
                full_response_chunks.append(str(chunk))
                response_chunk = {"type": "data", "content": chunk}
                yield f"data: {json.dumps(response_chunk, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.01) # 짧은 딜레이로 이벤트 스트림 안정화

            end_message = {"type": "end", "message": "Stream finished"}
            yield f"data: {json.dumps(end_message)}\n\n"

        except Exception as e:
            logger.error(f"스트리밍 중 오류 발생: {e}", exc_info=True)
            error_message = {"type": "error", "detail": f"스트리밍 중 오류가 발생했습니다: {str(e)}"}
            yield f"data: {json.dumps(error_message)}\n\n"
        finally:
            # ✨ 2. 스트림이 모두 끝난 후, 수집된 내용으로 DB 로그 업데이트
            final_text = "".join(full_response_chunks)
            if not final_text:
                logger.info("스트림 결과가 비어있어 로그를 업데이트하지 않습니다.")
                return

            try:
                logger.info(f"스트림 완료. Interaction ID [{workflow_req.interaction_id}]의 로그 업데이트 시작.")

                # 가장 최근에 생성된 로그 레코드를 찾습니다.
                log_to_update_list = db_manager.find_by_condition(
                    ExecutionIO,
                    {
                        "user_id": user_id,
                        "interaction_id": workflow_req.interaction_id,
                        # "workflow_id": workflow_req.workflow_id, # 워크플로우 ID 로직 삭제
                    },
                    limit=1,
                    orderby="created_at",
                    orderby_asc=False
                )

                if not log_to_update_list:
                    logger.warning(f"업데이트할 ExecutionIO 로그를 찾지 못했습니다. Interaction ID: {workflow_req.interaction_id}")
                    return

                log_to_update = log_to_update_list[0]

                # output_data 필드의 JSON을 실제 결과로 수정
                output_data_dict = json.loads(log_to_update.output_data)
                output_data_dict['result'] = final_text # placeholder를 최종 텍스트로 교체

                # inputs 필드에 있던 generator placeholder도 업데이트 (선택적)
                if 'inputs' in output_data_dict and isinstance(output_data_dict['inputs'], dict):
                    for key, value in output_data_dict['inputs'].items():
                        if value == "<generator_output>":
                            output_data_dict['inputs'][key] = final_text

                # 수정된 JSON으로 레코드를 업데이트
                log_to_update.output_data = json.dumps(output_data_dict, ensure_ascii=False)
                db_manager.update(log_to_update)

                logger.info(f"Interaction ID [{workflow_req.interaction_id}]의 로그가 최종 스트림 결과로 업데이트되었습니다.")

            except Exception as db_error:
                logger.error(f"ExecutionIO 로그 업데이트 중 DB 오류 발생: {db_error}", exc_info=True)
            finally:
                # 완료된 실행들 정리
                execution_manager.cleanup_completed_executions()


    try:
        user_id = extract_user_id_from_request(request)

        if request_body.workflow_name == 'default_mode':
            default_mode_workflow_folder = os.path.join(os.getcwd(), "constants")
            file_path = os.path.join(default_mode_workflow_folder, "base_chat_workflow.json")
            with open(file_path, 'r', encoding='utf-8') as f:
                workflow_data = json.load(f)
            workflow_data = await _default_workflow_parameter_helper(request, request_body, workflow_data)
        else:
            downloads_path = os.path.join(os.getcwd(), "downloads")
            download_path_id = os.path.join(downloads_path, user_id)
            filename = f"{request_body.workflow_name}.json" if not request_body.workflow_name.endswith('.json') else request_body.workflow_name
            file_path = os.path.join(download_path_id, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                workflow_data = json.load(f)
            workflow_data = await _workflow_parameter_helper(request_body, workflow_data)

        ## TODO 워크플로우 아이디 정합성 관련 로직 생각해볼 것
        # if workflow_data.get('workflow_id') != request_body.workflow_id:
        #     raise ValueError(f"워크플로우 ID가 일치하지 않습니다.")
        if not workflow_data or 'nodes' not in workflow_data or 'edges' not in workflow_data:
            raise ValueError(f"워크플로우 데이터가 유효하지 않습니다: {file_path}")

        if request_body.input_data is not None and request_body.input_data != "" and len(request_body.input_data) > 0:
            for node in workflow_data.get('nodes', []):
                if node.get('data', {}).get('functionId') == 'startnode':
                    parameters = node.get('data', {}).get('parameters', [])
                    if parameters:
                        parameters[0]['value'] = request_body.input_data
                        break

        app_db = get_db_manager(request)
        execution_meta = None
        if request_body.interaction_id != "default" and app_db:
            execution_meta = await get_or_create_execution_meta(
                app_db, user_id, request_body.interaction_id,
                request_body.workflow_id, request_body.workflow_name, request_body.input_data
            )

        if execution_meta:
            await update_execution_meta_count(app_db, execution_meta)

        # 비동기 실행기 생성
        executor = execution_manager.create_executor(
            workflow_data=workflow_data,
            db_manager=app_db,
            interaction_id=request_body.interaction_id,
            user_id=user_id
        )

        # 비동기 제너레이터 시작 (스트리밍용)
        result_generator = executor.execute_workflow_async_streaming()

        # StreamingResponse를 사용하여 SSE 스트림 반환
        return StreamingResponse(
            stream_generator(result_generator, app_db, user_id, request_body),
            media_type="text/event-stream"
        )

    except ValueError as e:
        logger.error(f"Workflow execution error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"An unexpected error occurred during workflow setup: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")


# # Helper Functions

# async def _workflow_parameter_helper(request_body: WorkflowRequest, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
#     """
#     Updates the workflow data by setting the interaction ID in the parameters of nodes
#     with a function ID of 'memory', if applicable.

#     Args:
#         request_body: An object containing the interaction ID to be applied.
#         workflow_data: A dictionary representing the workflow's nodes and their parameters.

#     Returns:
#         The updated workflow data with the interaction ID applied where necessary.
#     """
#     if (request_body.interaction_id) and (request_body.interaction_id != "default"):
#         for node in workflow_data.get('nodes', []):
#             if node.get('data', {}).get('functionId') == 'memory':
#                 parameters = node.get('data', {}).get('parameters', [])
#                 for parameter in parameters:
#                     if parameter.get('id') == 'interaction_id':
#                         parameter['value'] = request_body.interaction_id

#     if request_body.additional_params:
#         for node in workflow_data.get('nodes', []):
#             node_id = node.get('id')
#             if node_id and node_id in request_body.additional_params:
#                 node_params = request_body.additional_params[node_id]
#                 parameters = node.get('data', {}).get('parameters', [])

#                 # additional_params를 parameters에 추가
#                 additional_params = {
#                     "id": "additional_params",
#                     "name": "additional_params",
#                     "type": "DICT",
#                     "value": node_params
#                 }
#                 parameters.append(additional_params)

#     return workflow_data

# async def _default_workflow_parameter_helper(request, request_body: WorkflowRequest, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
#     """
#     Updates the workflow data with default parameters based on the request body.

#     This function modifies the `workflow_data` dictionary by setting specific parameter values
#     for nodes in the workflow. It updates the `collection_name` parameter for nodes with the
#     `document_loaders` function ID and the `interaction_id` parameter for nodes with the
#     `memory` function ID, based on the values provided in the `request_body`.

#     Parameters:
#         request_body: An object containing the request data. It should have attributes
#             `selected_collection` (optional) and `interaction_id` (optional).
#         workflow_data: A dictionary representing the workflow structure. It contains a list
#             of nodes, each of which may have a `data` dictionary with `functionId` and `parameters`.

#     Returns:
#         A dictionary representing the updated workflow data with modified parameters.
#     """
#     config_composer = request.app.state.config_composer
#     if not config_composer:
#         raise HTTPException(status_code=500, detail="Config composer not available")

#     llm_provider = config_composer.get_config_by_name("DEFAULT_LLM_PROVIDER").value

#     if llm_provider == "openai":
#         model = config_composer.get_config_by_name("OPENAI_MODEL_DEFAULT").value
#         url = config_composer.get_config_by_name("OPENAI_API_BASE_URL").value
#     elif llm_provider == "vllm":
#         model = config_composer.get_config_by_name("VLLM_MODEL_NAME").value
#         url = config_composer.get_config_by_name("VLLM_API_BASE_URL").value
#     else:
#         raise HTTPException(status_code=500, detail="Unsupported LLM provider")

#     for node in workflow_data.get('nodes', []):
#         if node.get('data', {}).get('functionId') == 'agents':
#             parameters = node.get('data', {}).get('parameters', [])
#             for parameter in parameters:
#                 if parameter.get('id') == 'model':
#                     parameter['value'] = model
#                 if parameter.get('id') == 'base_url':
#                     parameter['value'] = url

#     if request_body.selected_collections:
#         constant_folder = os.path.join(os.getcwd(), "constants")
#         collection_file_path = os.path.join(constant_folder, "collection_node_template.json")
#         edge_template_path = os.path.join(constant_folder, "base_edge_template.json")
#         with open(collection_file_path, 'r', encoding='utf-8') as f:
#             collection_node_template = json.load(f)
#         with open(edge_template_path, 'r', encoding='utf-8') as f:
#             edge_template = json.load(f)

#         for collection in request_body.selected_collections:
#             # UUID 부분을 제거하고 깨끗한 컬렉션 이름 추출
#             collection_name = extract_collection_name(collection)
#             coleection_code = hashlib.sha1(collection_name.encode('utf-8')).hexdigest()[:8]

#             print(f"Adding collection node for: {collection} (clean name: {collection_name})")
#             collection_node = copy.deepcopy(collection_node_template)
#             edge = copy.deepcopy(edge_template)
#             collection_node['id'] = f"document_loaders_{collection}"
#             collection_node['data']['parameters'][0]['value'] = collection # collection에서 collection_name으로 수정함. uuid 제거 위해서 수정
#             collection_node['data']['parameters'][1]['value'] = f"retrieval_search_tool_for_{coleection_code}"
#             collection_node['data']['parameters'][2]['value'] = f"Use when a search is needed for the given question related to {collection_name}."
#             workflow_data['nodes'].append(collection_node)

#             edge_id = f"{collection_node['id']}:tools-default_agents:tools-{coleection_code}"
#             edge['id'] = edge_id
#             edge['source']['nodeId'] = collection_node['id']
#             workflow_data['edges'].append(edge)

#     if (request_body.interaction_id) and (request_body.interaction_id != "default"):
#         for node in workflow_data.get('nodes', []):
#             if node.get('data', {}).get('functionId') == 'memory':
#                 parameters = node.get('data', {}).get('parameters', [])
#                 for parameter in parameters:
#                     if parameter.get('id') == 'interaction_id':
#                         parameter['value'] = request_body.interaction_id

#     if request_body.additional_params:
#         for node in workflow_data.get('nodes', []):
#             node_id = node.get('id')
#             if node_id and node_id in request_body.additional_params:
#                 node_params = request_body.additional_params[node_id]
#                 parameters = node.get('data', {}).get('parameters', [])

#                 # additional_params를 parameters에 추가
#                 additional_param = {
#                     "id": "additional_params",
#                     "name": "additional_params",
#                     "type": "DICT",
#                     "value": node_params
#                 }
#                 parameters.append(additional_param)

#     return workflow_data


# ==================================================
# 기존 모델들 뒤에 추가할 새로운 Pydantic 모델들
# ==================================================

class BatchTestCase(BaseModel):
    """배치 테스트 케이스 모델"""
    id: int
    input: str
    expected_output: Optional[str] = None

class BatchExecuteRequest(BaseModel):
    """배치 실행 요청 모델"""
    workflow_name: str
    workflow_id: str
    test_cases: List[BatchTestCase]
    batch_size: int = 5
    interaction_id: str = "batch_test"
    selected_collections: Optional[List[str]] = None

class BatchTestResult(BaseModel):
    """배치 테스트 결과 모델"""
    id: int
    input: str
    expected_output: Optional[str]
    actual_output: Optional[str]
    status: str  # 'success', 'error'
    execution_time: Optional[int]  # milliseconds
    error: Optional[str]

class BatchExecuteResponse(BaseModel):
    """배치 실행 응답 모델"""
    batch_id: str
    total_count: int
    success_count: int
    error_count: int
    total_execution_time: int
    results: List[BatchTestResult]

# 배치 작업 상태 저장용 (메모리 기반)
batch_status_storage = {}

# 배치 처리 함수들
async def execute_single_workflow_for_batch(
    user_id: str,
    workflow_name: str,
    workflow_id: str,
    input_data: str,
    interaction_id: str,
    selected_collections: Optional[List[str]],
    app_db,
    request
) -> Dict[str, Any]:
    """
    배치 처리를 위한 단일 워크플로우 실행 함수
    기존 execute_workflow_with_id 로직을 재사용하되 Generator 처리 추가
    """
    start_time = time.time()

    try:
        # ========== 워크플로우 데이터 로드 ==========
        if workflow_name == 'default_mode':
            # 기본 채팅 모드 처리
            default_mode_workflow_folder = os.path.join(os.getcwd(), "constants")
            file_path = os.path.join(default_mode_workflow_folder, "base_chat_workflow.json")
            with open(file_path, 'r', encoding='utf-8') as f:
                workflow_data = json.load(f)

            # WorkflowRequest 객체 생성 (기존 헬퍼 함수 호환)
            temp_request = WorkflowRequest(
                workflow_name=workflow_name,
                workflow_id=workflow_id,
                input_data=input_data,
                interaction_id=interaction_id,
                selected_collections=selected_collections
            )
            workflow_data = await _default_workflow_parameter_helper(request, temp_request, workflow_data)
        else:
            # 사용자 정의 워크플로우 처리
            downloads_path = os.path.join(os.getcwd(), "downloads")
            download_path_id = os.path.join(downloads_path, user_id)

            filename = f"{workflow_name}.json" if not workflow_name.endswith('.json') else workflow_name
            file_path = os.path.join(download_path_id, filename)

            if not os.path.exists(file_path):
                raise ValueError(f"워크플로우 파일을 찾을 수 없습니다: {file_path}")

            with open(file_path, 'r', encoding='utf-8') as f:
                workflow_data = json.load(f)

            temp_request = WorkflowRequest(
                workflow_name=workflow_name,
                workflow_id=workflow_id,
                input_data=input_data,
                interaction_id=interaction_id,
                selected_collections=selected_collections
            )
            workflow_data = await _workflow_parameter_helper(temp_request, workflow_data)

        # ========== 워크플로우 데이터 검증 ==========
        # if workflow_data.get('workflow_id') != workflow_id:
        #     raise ValueError(f"워크플로우 ID가 일치하지 않습니다: {workflow_data.get('workflow_id')} != {workflow_id}")

        if not workflow_data or 'nodes' not in workflow_data or 'edges' not in workflow_data:
            raise ValueError(f"워크플로우 데이터가 유효하지 않습니다")

        # ========== 입력 데이터 설정 ==========
        if input_data is not None:
            for node in workflow_data.get('nodes', []):
                if node.get('data', {}).get('functionId') == 'startnode':
                    parameters = node.get('data', {}).get('parameters', [])
                    if parameters and isinstance(parameters, list):
                        parameters[0]['value'] = input_data
                        break

        # ========== 워크플로우 실행 ==========
        executor = execution_manager.create_executor(
            workflow_data=workflow_data,
            db_manager=app_db,
            interaction_id=interaction_id,
            user_id=user_id
        )

        # 비동기 실행 및 결과 수집
        final_outputs = []
        async for chunk in executor.execute_workflow_async():
            final_outputs.append(chunk)

        # ========== 결과 처리 ==========
        if len(final_outputs) == 1:
            processed_output = final_outputs[0]
        elif len(final_outputs) > 1:
            # 스트리밍인 경우 모든 청크를 합치거나 마지막 값 사용
            if all(isinstance(item, str) for item in final_outputs):
                processed_output = ''.join(final_outputs)  # 문자열이면 연결
            else:
                processed_output = final_outputs[-1]  # 아니면 마지막 값
        else:
            processed_output = "결과 없음"

        execution_time = int((time.time() - start_time) * 1000)

        return {
            "success": True,
            "outputs": processed_output,
            "execution_time": execution_time
        }

    except Exception as e:
        execution_time = int((time.time() - start_time) * 1000)
        logger.error(f"배치 워크플로우 실행 실패: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "execution_time": execution_time
        }

async def process_batch_group(
    user_id: str,
    workflow_name: str,
    workflow_id: str,
    test_cases: List[BatchTestCase],
    interaction_id: str,
    selected_collections: Optional[List[str]],
    batch_id: str,
    app_db,
    request
) -> List[BatchTestResult]:
    """
    배치 그룹을 병렬로 처리
    """
    results = []

    # asyncio.gather를 사용해서 병렬 실행
    tasks = []
    for test_case in test_cases:
        unique_interaction_id = f"{interaction_id}_{batch_id}_{test_case.id}"
        task = execute_single_workflow_for_batch(
            user_id=user_id,
            workflow_name=workflow_name,
            workflow_id=workflow_id,
            input_data=test_case.input,
            interaction_id=unique_interaction_id,
            selected_collections=selected_collections,
            app_db=app_db,
            request=request
        )
        tasks.append(task)

    # 모든 태스크를 병렬로 실행
    execution_results = await asyncio.gather(*tasks, return_exceptions=True)

    # 결과 처리
    for i, (test_case, exec_result) in enumerate(zip(test_cases, execution_results)):
        if isinstance(exec_result, Exception):
            result = BatchTestResult(
                id=test_case.id,
                input=test_case.input,
                expected_output=test_case.expected_output,
                actual_output=None,
                status="error",
                execution_time=0,
                error=str(exec_result)
            )
        elif exec_result.get("success"):
            # outputs 처리 - 다양한 형태의 결과를 문자열로 변환
            outputs = exec_result.get("outputs", "결과 없음")
            if isinstance(outputs, list):
                actual_output = outputs[0] if outputs else "결과 없음"
            else:
                actual_output = str(outputs)

            result = BatchTestResult(
                id=test_case.id,
                input=test_case.input,
                expected_output=test_case.expected_output,
                actual_output=actual_output,
                status="success",
                execution_time=exec_result.get("execution_time", 0),
                error=None
            )
        else:
            result = BatchTestResult(
                id=test_case.id,
                input=test_case.input,
                expected_output=test_case.expected_output,
                actual_output=None,
                status="error",
                execution_time=exec_result.get("execution_time", 0),
                error=exec_result.get("error", "알 수 없는 오류")
            )

        results.append(result)

        # 진행 상황 업데이트
        if batch_id in batch_status_storage:
            batch_status_storage[batch_id]["completed_count"] += 1
            progress = (batch_status_storage[batch_id]["completed_count"] /
                       batch_status_storage[batch_id]["total_count"]) * 100
            batch_status_storage[batch_id]["progress"] = progress

    return results

# ==================================================
# 배치 실행 API 엔드포인트 / batch로 만듦.
# ==================================================

@router.post("/execute/batch", response_model=BatchExecuteResponse)
async def execute_workflow_batch(request: Request, batch_request: BatchExecuteRequest):
    """
    워크플로우 배치 실행 엔드포인트
    여러 테스트 케이스를 배치로 처리하여 서버 부하를 줄입니다.

    Args:
        batch_request: 배치 실행 요청 데이터

    Returns:
        BatchExecuteResponse: 배치 실행 결과
    """
    try:
        user_id = extract_user_id_from_request(request)
        app_db = get_db_manager(request)

        batch_id = str(uuid.uuid4())
        start_time = time.time()

        # 배치 상태 초기화
        batch_status_storage[batch_id] = {
            "status": "running",
            "total_count": len(batch_request.test_cases),
            "completed_count": 0,
            "progress": 0.0,
            "start_time": start_time
        }

        logger.info(f"배치 {batch_id} 시작: 워크플로우={batch_request.workflow_name}, "
                   f"테스트 케이스={len(batch_request.test_cases)}개, 배치 크기={batch_request.batch_size}")

        all_results = []

        # 배치 크기만큼 나누어서 처리
        for i in range(0, len(batch_request.test_cases), batch_request.batch_size):
            batch_group = batch_request.test_cases[i:i + batch_request.batch_size]

            logger.info(f"배치 그룹 {i//batch_request.batch_size + 1} 처리 중: {len(batch_group)}개 병렬 실행")

            # 현재 배치 그룹 처리
            group_results = await process_batch_group(
                user_id=user_id,
                workflow_name=batch_request.workflow_name,
                workflow_id=batch_request.workflow_id,
                test_cases=batch_group,
                interaction_id=batch_request.interaction_id,
                selected_collections=batch_request.selected_collections,
                batch_id=batch_id,
                app_db=app_db,
                request=request
            )

            all_results.extend(group_results)

            # 다음 배치 그룹 처리 전 잠시 대기 (서버 부하 방지위해서)
            if i + batch_request.batch_size < len(batch_request.test_cases):
                await asyncio.sleep(0.5)

        # 최종 결과 계산
        total_execution_time = int((time.time() - start_time) * 1000)
        success_count = sum(1 for r in all_results if r.status == "success")
        error_count = len(all_results) - success_count

        # 배치 상태 완료로 업데이트
        batch_status_storage[batch_id]["status"] = "completed"
        batch_status_storage[batch_id]["progress"] = 100.0

        logger.info(f"배치 {batch_id} 완료: 성공={success_count}개, 실패={error_count}개, "
                   f"총 소요시간={total_execution_time}ms")

        response = BatchExecuteResponse(
            batch_id=batch_id,
            total_count=len(all_results),
            success_count=success_count,
            error_count=error_count,
            total_execution_time=total_execution_time,
            results=all_results
        )

        return response

    except Exception as e:
        logger.error(f"배치 실행 중 오류: {str(e)}", exc_info=True)

        if 'batch_id' in locals() and batch_id in batch_status_storage:
            batch_status_storage[batch_id]["status"] = "error"
            batch_status_storage[batch_id]["error"] = str(e)

        raise HTTPException(status_code=500, detail=f"배치 실행 실패: {str(e)}")

@router.get("/batch/status/{batch_id}")
async def get_batch_status(batch_id: str):
    """
    배치 실행 상태 조회 (선택사항 - 실시간 진행 상황 확인용)

    Args:
        batch_id: 배치 작업 ID

    Returns:
        배치 작업 상태 정보
    """
    if batch_id not in batch_status_storage:
        raise HTTPException(status_code=404, detail="배치를 찾을 수 없습니다")

    return JSONResponse(content=batch_status_storage[batch_id])

@router.get("/execution/status")
async def get_all_execution_status():
    """
    현재 실행 중인 모든 워크플로우의 상태를 반환합니다.

    Returns:
        모든 활성 워크플로우 실행 상태
    """
    try:
        status_data = execution_manager.get_all_execution_status()
        return JSONResponse(content={
            "active_executions": len(status_data),
            "executions": status_data
        })
    except Exception as e:
        logger.error(f"실행 상태 조회 중 오류: {e}")
        raise HTTPException(status_code=500, detail="실행 상태 조회 실패")

@router.get("/execution/status/{execution_id}")
async def get_execution_status(execution_id: str):
    """
    특정 워크플로우 실행의 상태를 반환합니다.

    Args:
        execution_id: 실행 ID (interaction_id_workflow_id_user_id 형태)

    Returns:
        워크플로우 실행 상태
    """
    try:
        status = execution_manager.get_execution_status(execution_id)
        if status is None:
            raise HTTPException(status_code=404, detail="실행을 찾을 수 없습니다")
        return JSONResponse(content=status)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"실행 상태 조회 중 오류: {e}")
        raise HTTPException(status_code=500, detail="실행 상태 조회 실패")

@router.post("/execution/cleanup")
async def cleanup_completed_executions():
    """
    완료된 워크플로우 실행들을 정리합니다.

    Returns:
        정리 결과
    """
    try:
        before_count = len(execution_manager.get_all_execution_status())
        execution_manager.cleanup_completed_executions()
        after_count = len(execution_manager.get_all_execution_status())
        cleaned_count = before_count - after_count

        return JSONResponse(content={
            "message": f"{cleaned_count}개의 완료된 실행이 정리되었습니다",
            "before_count": before_count,
            "after_count": after_count,
            "cleaned_count": cleaned_count
        })
    except Exception as e:
        logger.error(f"실행 정리 중 오류: {e}")
        raise HTTPException(status_code=500, detail="실행 정리 실패")

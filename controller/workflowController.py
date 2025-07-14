from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import os
import json
import logging
from datetime import datetime
from src.node_composer import get_node_registry, get_node_class_registry
from src.workflow_executor import WorkflowExecutor

logger = logging.getLogger("workflow-controller")
router = APIRouter(prefix="/workflow", tags=["workflow"])

class WorkflowData(BaseModel):
    workflow_name: str
    workflow_id: str
    view: Dict[str, Any]
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]

class SaveWorkflowRequest(BaseModel):
    workflow_id: str
    content: WorkflowData

@router.get("/list")
async def list_workflows():
    """
    downloads 폴더에 있는 모든 workflow 파일들의 이름을 반환합니다.
    """
    try:
        downloads_path = os.path.join(os.getcwd(), "downloads")
        
        # downloads 폴더가 존재하지 않으면 생성
        if not os.path.exists(downloads_path):
            os.makedirs(downloads_path)
            return JSONResponse(content={"workflows": []})
        
        # .json 확장자를 가진 파일들만 필터링
        workflow_files = []
        for file in os.listdir(downloads_path):
            if file.endswith('.json'):
                workflow_files.append(file)
        
        logger.info(f"Found {len(workflow_files)} workflow files")
        return JSONResponse(content={"workflows": workflow_files})
        
    except Exception as e:
        logger.error(f"Error listing workflows: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list workflows: {str(e)}")

@router.post("/save")
async def save_workflow(request: SaveWorkflowRequest):
    """
    Frontend에서 받은 workflow 정보를 파일로 저장합니다.
    파일명: {workflow_id}.json
    """
    try:
        downloads_path = os.path.join(os.getcwd(), "downloads")
        
        # downloads 폴더가 존재하지 않으면 생성
        if not os.path.exists(downloads_path):
            os.makedirs(downloads_path)
        
        # 파일명 생성 (workflow_id + .json)
        filename = f"{request.workflow_id}.json"
        file_path = os.path.join(downloads_path, filename)
        
        # workflow content를 JSON 파일로 저장
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(request.content.dict(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"Workflow saved successfully: {filename}")
        return JSONResponse(content={
            "success": True,
            "message": f"Workflow '{request.workflow_id}' saved successfully",
            "filename": filename
        })
        
    except Exception as e:
        logger.error(f"Error saving workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save workflow: {str(e)}")

@router.get("/load/{workflow_id}")
async def load_workflow(workflow_id: str):
    """
    특정 workflow를 로드합니다.
    """
    try:
        downloads_path = os.path.join(os.getcwd(), "downloads")
        filename = f"{workflow_id}.json"
        file_path = os.path.join(downloads_path, filename)
        
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

@router.delete("/delete/{workflow_id}")
async def delete_workflow(workflow_id: str):
    """
    특정 workflow를 삭제합니다.
    """
    try:
        downloads_path = os.path.join(os.getcwd(), "downloads")
        filename = f"{workflow_id}.json"
        file_path = os.path.join(downloads_path, filename)
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"Workflow '{workflow_id}' not found")
        
        os.remove(file_path)
        
        logger.info(f"Workflow deleted successfully: {filename}")
        return JSONResponse(content={
            "success": True,
            "message": f"Workflow '{workflow_id}' deleted successfully"
        })
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Workflow '{workflow_id}' not found")
    except Exception as e:
        logger.error(f"Error deleting workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete workflow: {str(e)}")

@router.get("/list/detail")
async def list_workflows_detail():
    """
    downloads 폴더에 있는 모든 workflow 파일들의 상세 정보를 반환합니다.
    각 워크플로우에 대해 파일명, workflow_id, 노드 수, 마지막 수정일자를 포함합니다.
    """
    try:
        downloads_path = os.path.join(os.getcwd(), "downloads")
        
        # downloads 폴더가 존재하지 않으면 생성
        if not os.path.exists(downloads_path):
            os.makedirs(downloads_path)
            return JSONResponse(content={"workflows": []})
        
        workflow_details = []
        
        # .json 확장자를 가진 파일들만 처리
        for file in os.listdir(downloads_path):
            if not file.endswith('.json'):
                continue
                
            file_path = os.path.join(downloads_path, file)
            
            try:
                # 파일 메타데이터 수집
                file_stat = os.stat(file_path)
                last_modified = datetime.fromtimestamp(file_stat.st_mtime).isoformat()
                
                # JSON 파일 읽기
                with open(file_path, 'r', encoding='utf-8') as f:
                    workflow_data = json.load(f)
                
                # workflow_id 추출 (workflow_id 필드)
                workflow_id = workflow_data.get('workflow_id', 'unknown')
                
                # nodes 수 계산
                nodes = workflow_data.get('nodes', [])
                node_count = len(nodes) if isinstance(nodes, list) else 0
                
                # 상세 정보 추가
                workflow_detail = {
                    "filename": file,
                    "workflow_id": workflow_id,
                    "node_count": node_count,
                    "last_modified": last_modified
                }
                
                workflow_details.append(workflow_detail)
                logger.debug(f"Processed workflow file: {file} (ID: {workflow_id}, Nodes: {node_count})")
                
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON file {file}: {str(e)}")
                # 파싱 실패한 파일도 기본 정보로 포함
                workflow_details.append({
                    "filename": file,
                    "workflow_id": "invalid_json",
                    "node_count": 0,
                    "last_modified": datetime.fromtimestamp(os.stat(file_path).st_mtime).isoformat(),
                    "error": "Invalid JSON format"
                })
            except Exception as e:
                logger.warning(f"Failed to process workflow file {file}: {str(e)}")
                # 처리 실패한 파일도 기본 정보로 포함
                workflow_details.append({
                    "filename": file,
                    "workflow_id": "error",
                    "node_count": 0,
                    "last_modified": datetime.fromtimestamp(os.stat(file_path).st_mtime).isoformat(),
                    "error": str(e)
                })
        
        logger.info(f"Found {len(workflow_details)} workflow files with detailed information")
        return JSONResponse(content={"workflows": workflow_details})
        
    except Exception as e:
        logger.error(f"Error listing workflow details: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list workflow details: {str(e)}")

@router.post("/execute", response_model=Dict[str, Any])
async def execute_workflow(request: Request, workflow: WorkflowData):
    """
    주어진 노드와 엣지 정보로 워크플로우를 실행합니다.
    """
    
    # print("DEBUG: 워크플로우 실행 요청\n", workflow)
    
    try:
        workflow_data = workflow.dict()
        
        # 데이터베이스 매니저 가져오기
        db_manager = None
        if hasattr(request.app.state, 'app_db') and request.app.state.app_db:
            db_manager = request.app.state.app_db
        
        executor = WorkflowExecutor(workflow_data, db_manager)
        final_outputs = executor.execute_workflow()
        
        return {"status": "success", "message": "워크플로우 실행 완료", "outputs": final_outputs}

    except ValueError as e:
        logging.error(f"Workflow execution error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

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
        # 데이터베이스 매니저 가져오기
        if not hasattr(request.app.state, 'app_db') or not request.app.state.app_db:
            raise HTTPException(status_code=500, detail="Database connection not available")
        
        db_manager = request.app.state.app_db
        
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
        WHERE workflow_name = %s AND workflow_id = %s
        GROUP BY node_id, node_name
        ORDER BY node_id
        """
        
        # SQLite인 경우 파라미터 플레이스홀더 변경
        if db_manager.config_db_manager.db_type == "sqlite":
            query = query.replace("%s", "?")
        
        # 쿼리 실행
        result = db_manager.config_db_manager.execute_query(query, (workflow_name, workflow_id))
        
        if not result:
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
            # Decimal 타입을 float로 변환하는 헬퍼 함수
            def safe_float(value):
                if value is None:
                    return None
                return float(value)
            
            def safe_round_float(value, decimals=2):
                if value is None:
                    return None
                return round(float(value), decimals)
            
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
        avg_total_processing_time = sum(stat['avg_processing_time_ms'] * stat['execution_count'] for stat in performance_stats) / total_executions if total_executions > 0 else 0.0
        avg_total_cpu_usage = sum(stat['avg_cpu_usage_percent'] * stat['execution_count'] for stat in performance_stats) / total_executions if total_executions > 0 else 0.0
        avg_total_ram_usage = sum(stat['avg_ram_usage_mb'] * stat['execution_count'] for stat in performance_stats) / total_executions if total_executions > 0 else 0.0
        
        # GPU 통계 (GPU 데이터가 있는 경우만)
        gpu_stats = None
        total_gpu_executions = sum(stat['gpu_execution_count'] for stat in performance_stats)
        if total_gpu_executions > 0:
            gpu_processing_time_sum = sum(stat['avg_processing_time_ms'] * stat['gpu_execution_count'] for stat in performance_stats if stat['avg_gpu_usage_percent'] is not None)
            gpu_usage_sum = sum(stat['avg_gpu_usage_percent'] * stat['gpu_execution_count'] for stat in performance_stats if stat['avg_gpu_usage_percent'] is not None)
            gpu_memory_sum = sum(stat['avg_gpu_memory_mb'] * stat['gpu_execution_count'] for stat in performance_stats if stat['avg_gpu_memory_mb'] is not None)
            
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
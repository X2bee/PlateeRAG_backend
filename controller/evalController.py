from fastapi import APIRouter, HTTPException, Query, BackgroundTasks, Request
from pydantic import BaseModel, Field
from typing import Dict, Optional, Any, List
from service.eval.evaluator import (
    evaluator_bert, 
    evaluator_LM, 
    HfCheck, 
    evaluator_LM_task
)
import logging
import datetime
import json
import os
import threading
import uuid
from controller.controller_helper import extract_user_id_from_request

# 로깅 설정
logger = logging.getLogger("polar-evaluator")

# 평가 작업 데이터를 저장할 디렉터리
EVAL_JOB_DATA_DIR = "eval_job_data"

# 디렉터리가 없으면 생성
os.makedirs(EVAL_JOB_DATA_DIR, exist_ok=True)

router = APIRouter(
    prefix="/api/eval",
    tags=["evaluation"],
    responses={404: {"description": "Not found"}},
)

# 요청 모델 정의
class EvalRequest(BaseModel):
    job_name: str = Field(..., description="Job name")
    task: str = Field(..., description="Task type")
    model_name: str = Field(..., description="Model name or path")
    dataset_name: str = Field(..., description="Dataset name or path")
    column1: Optional[str] = Field(None, description="Input column 1")
    column2: Optional[str] = Field(None, description="Input column 2")
    column3: Optional[str] = Field(None, description="Input column 3")
    label: Optional[str] = Field(None, description="Label column")
    top_k: Optional[int] = Field(1, description="Top K value")
    gpu_num: Optional[int] = Field(1, description="Number of GPUs")
    model_minio_enabled: bool = Field(..., description="Enable Minio for model")
    dataset_minio_enabled: bool = Field(..., description="Enable Minio for dataset")
    use_cot: Optional[bool] = Field(False, description="Use Chain of Thought")
    base_model: Optional[str] = Field(None, description="Base model for comparison")

# 응답 모델 정의
class EvalResponse(BaseModel):
    job_id: str
    status: str
    message: str

class EvalStatusResponse(BaseModel):
    job_id: str
    status: str
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    job_info: Dict[str, Any]
    result: Optional[Dict[str, Any]] = None
    base_model_result: Optional[Dict[str, Any]] = None
    base_model_name: Optional[str] = None

# 실행 중인 평가 작업을 추적하기 위한 딕셔너리
evaluation_jobs = {}
jobs_lock = threading.Lock()

def get_eval_job_path(job_id: str, user_id: str = None) -> str:
    """평가 작업 파일 경로를 반환합니다."""
    if user_id:
        user_dir = os.path.join(EVAL_JOB_DATA_DIR, user_id)
        os.makedirs(user_dir, exist_ok=True)
        return os.path.join(user_dir, f"{job_id}.json")
    return os.path.join(EVAL_JOB_DATA_DIR, f"{job_id}.json")

def save_eval_job_to_json(job_id: str, job_data: Dict[str, Any], user_id: str = None) -> None:
    """평가 작업 데이터를 JSON 파일로 저장합니다."""
    try:
        file_path = get_eval_job_path(job_id, user_id)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(job_data, f, ensure_ascii=False, indent=4)
        logger.info(f"Evaluation job data saved to {file_path}")
    except Exception as e:
        logger.error(f"Error saving evaluation job data: {e}")
        raise HTTPException(status_code=500, detail="Failed to save evaluation job data")

def load_eval_job_from_json(job_id: str, user_id: str = None) -> Dict[str, Any]:
    """JSON 파일에서 평가 작업 데이터를 로드합니다."""
    try:
        file_path = get_eval_job_path(job_id, user_id)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Evaluation job not found")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading evaluation job: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

def load_all_eval_jobs_from_json(user_id: str = None) -> Dict[str, Dict[str, Any]]:
    """모든 평가 작업 데이터를 JSON 파일에서 로드합니다."""
    try:
        jobs = {}
        base_dir = os.path.join(EVAL_JOB_DATA_DIR, user_id) if user_id else EVAL_JOB_DATA_DIR
        
        if not os.path.exists(base_dir):
            return jobs
        
        job_files = [f for f in os.listdir(base_dir) if f.endswith('.json')]
        
        for file_name in job_files:
            try:
                job_id = file_name.replace('.json', '')
                file_path = os.path.join(base_dir, file_name)
                with open(file_path, 'r', encoding='utf-8') as f:
                    jobs[job_id] = json.load(f)
            except Exception as e:
                logger.error(f"Error loading job data from {file_name}: {e}")
        
        return jobs
    except Exception as e:
        logger.error(f"Error loading all evaluation jobs: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

def run_evaluation_thread(request: EvalRequest, job_id: str, user_id: str = None):
    """평가 작업을 실행하는 스레드 함수"""
    log_filename = get_eval_job_path(job_id, user_id)
    
    # 결과 변수들 초기화
    final_result = None
    base_model_result = None
    base_model_name = None
    
    # 초기 데이터 설정
    initial_data = {
        "job_info": request.model_dump(),
        "logs": [],
        "status": "running",
        "start_time": datetime.datetime.now().isoformat()
    }
    
    try:
        with open(log_filename, "w", encoding="utf-8") as f:
            json.dump(initial_data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        logger.error(f"Failed to create initial job file: {e}")
        return
    
    def update_log_file(log_entry):
        """로그 파일을 업데이트합니다."""
        try:
            with open(log_filename, "r+", encoding="utf-8") as f:
                data = json.load(f)
                if "logs" not in data:
                    data["logs"] = []
                data["logs"].append(log_entry)
                f.seek(0)
                json.dump(data, f, ensure_ascii=False, indent=4)
                f.truncate()
        except Exception as e:
            logger.error(f"Failed to update log file: {e}")
    
    class CaptureHandler(logging.Handler):
        def emit(self, record):
            try:
                timestamp = datetime.datetime.fromtimestamp(record.created).isoformat()
                log_entry = {
                    "timestamp": timestamp,
                    "level": record.levelname,
                    "message": record.getMessage()
                }
                update_log_file(log_entry)
            except Exception:
                pass
    
    capture_handler = CaptureHandler()
    capture_handler.setLevel(logging.DEBUG)
    root_logger = logging.getLogger()
    root_logger.addHandler(capture_handler)
    
    try:
        # 작업 상태 업데이트
        with jobs_lock:
            job_data = {
                "status": "running",
                "start_time": datetime.datetime.now().isoformat(),
                "job_info": request.model_dump()
            }
            evaluation_jobs[job_id] = job_data
            
            # JSON 파일 업데이트
            with open(log_filename, "r+", encoding="utf-8") as f:
                data = json.load(f)
                data.update(job_data)
                f.seek(0)
                json.dump(data, f, ensure_ascii=False, indent=4)
                f.truncate()
        
        logger.info(f"Starting evaluation job {job_id}")
        logger.info(f"Parameters: {request.model_dump()}")
        
        # 평가 로직 실행
        if request.task == 'CausalLM':
            logger.info(f"Running CausalLM evaluation")
            result = evaluator_LM(
                model_name=request.model_name,
                dataset_name=request.dataset_name,
                gpu_count=request.gpu_num,
                use_cot=request.use_cot,
                column1=request.column1,
                column2=request.column2,
                column3=request.column3,
                label=request.label,
                model_minio_enabled=request.model_minio_enabled,
                dataset_minio_enabled=request.dataset_minio_enabled
            )
            final_result = result
            logger.info(f"CausalLM evaluation completed")
            
        elif request.task == "CausalLM_task":
            logger.info(f"Running CausalLM_task evaluation")
            
            has_base_model = request.base_model is not None and request.base_model.strip() != ""
            
            if has_base_model:
                logger.info(f"Base model specified: {request.base_model}")
                base_model_name = request.base_model
                
                main_result, base_result = evaluator_LM_task(
                    job_name=job_id,
                    model_name=request.model_name,
                    dataset_name=request.dataset_name,
                    gpu_count=request.gpu_num,
                    model_minio_enabled=request.model_minio_enabled,
                    base_model=request.base_model
                )
                
                final_result = main_result
                base_model_result = base_result
                
                logger.info(f"Main model evaluation completed")
                logger.info(f"Base model evaluation completed")
                
            else:
                logger.info("Running single model evaluation")
                result, _ = evaluator_LM_task(
                    job_name=job_id,
                    model_name=request.model_name,
                    dataset_name=request.dataset_name,
                    gpu_count=request.gpu_num,
                    model_minio_enabled=request.model_minio_enabled,
                )
                final_result = result
                logger.info(f"Single model evaluation completed")
            
            # 에러 체크
            if final_result and 'error' in final_result:
                error_message = final_result.get('error', 'Unknown error in main model')
                logger.error(f"Main model evaluation failed: {error_message}")
                raise Exception(error_message)
            
            if base_model_result and 'error' in base_model_result:
                error_message = base_model_result.get('error', 'Unknown error in base model')
                logger.error(f"Base model evaluation failed: {error_message}")
                logger.warning(f"Base model evaluation failed but continuing")
                base_model_result = {"error": error_message}
                
        else:
            logger.info(f"Running {request.task} evaluation")
            result = evaluator_bert(
                task=request.task,
                model_name=request.model_name,
                dataset_name=request.dataset_name,
                column1=request.column1,
                column2=request.column2,
                column3=request.column3,
                label=request.label,
                top_k=request.top_k,
                gpu_num=request.gpu_num,
                model_minio_enabled=request.model_minio_enabled,
                dataset_minio_enabled=request.dataset_minio_enabled
            )
            final_result = result
            logger.info(f"{request.task} evaluation completed")
        
        # 작업 완료 상태 업데이트
        with jobs_lock:
            evaluation_jobs[job_id]["status"] = "completed"
            evaluation_jobs[job_id]["end_time"] = datetime.datetime.now().isoformat()
            evaluation_jobs[job_id]["result"] = final_result
            
            if base_model_result is not None:
                evaluation_jobs[job_id]["base_model_result"] = base_model_result
                evaluation_jobs[job_id]["base_model_name"] = base_model_name
            
            # JSON 파일 업데이트
            with open(log_filename, "r+", encoding="utf-8") as f:
                data = json.load(f)
                data["status"] = "completed"
                data["end_time"] = datetime.datetime.now().isoformat()
                data["result"] = final_result
                
                if base_model_result is not None:
                    data["base_model_result"] = base_model_result
                    data["base_model_name"] = base_model_name
                
                f.seek(0)
                json.dump(data, f, ensure_ascii=False, indent=4)
                f.truncate()
        
        logger.info(f"Evaluation job {job_id} completed successfully")
        return final_result
    
    except Exception as e:
        error_message = f"Evaluation job {job_id} failed: {str(e)}"
        logger.error(error_message)
        
        # 실패 상태 업데이트
        with jobs_lock:
            evaluation_jobs[job_id]["status"] = "failed"
            evaluation_jobs[job_id]["error"] = str(e)
            evaluation_jobs[job_id]["end_time"] = datetime.datetime.now().isoformat()
            
            # JSON 파일 업데이트
            try:
                with open(log_filename, "r+", encoding="utf-8") as f:
                    data = json.load(f)
                    data["status"] = "failed"
                    data["error"] = str(e)
                    data["end_time"] = datetime.datetime.now().isoformat()
                    f.seek(0)
                    json.dump(data, f, ensure_ascii=False, indent=4)
                    f.truncate()
            except Exception:
                pass
        
        raise HTTPException(status_code=500, detail=error_message)
    
    finally:
        root_logger.removeHandler(capture_handler)
        capture_handler.close()

def run_evaluation(request: EvalRequest, job_id: str, user_id: str = None):
    """평가 작업을 실행하는 스레드를 시작합니다."""
    try:
        thread = threading.Thread(
            target=run_evaluation_thread,
            args=(request, job_id, user_id),
            daemon=True
        )
        thread.start()
        logger.info(f"Started evaluation thread for job {job_id}")
    except Exception as e:
        logger.error(f"Error starting evaluation thread: {e}")
        raise HTTPException(status_code=500, detail="Failed to start evaluation job")

@router.post("", response_model=EvalResponse)
async def create_evaluation(request: EvalRequest, background_tasks: BackgroundTasks, req: Request):
    """평가 작업을 비동기적으로 시작합니다."""
    try:
        user_id = extract_user_id_from_request(req)
        job_id = f"{request.job_name}_{uuid.uuid4().hex[:8]}"
        
        background_tasks.add_task(run_evaluation, request, job_id, user_id)
        
        return EvalResponse(
            job_id=job_id,
            status="accepted",
            message="Evaluation job started successfully"
        )
    except Exception as e:
        logger.error(f"Error creating evaluation job: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@router.get("", response_model=Dict[str, Dict[str, Any]])
async def get_all_evaluations(request: Request):
    """모든 평가 작업의 상태를 조회합니다."""
    try:
        user_id = extract_user_id_from_request(request)
        return load_all_eval_jobs_from_json(user_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting all evaluation jobs: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@router.get("/{job_id}", response_model=EvalStatusResponse)
async def get_evaluation_status(job_id: str, request: Request):
    """특정 평가 작업의 상태를 조회합니다."""
    try:
        user_id = extract_user_id_from_request(request)
        job_data = load_eval_job_from_json(job_id, user_id)
        
        return EvalStatusResponse(
            job_id=job_id,
            status=job_data.get("status", "unknown"),
            start_time=job_data.get("start_time"),
            end_time=job_data.get("end_time"),
            job_info=job_data.get("job_info", {}),
            result=job_data.get("result"),
            base_model_result=job_data.get("base_model_result"),
            base_model_name=job_data.get("base_model_name")
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting evaluation status: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@router.delete("/{job_id}")
async def delete_evaluation_job(job_id: str, request: Request):
    """특정 평가 작업을 삭제합니다."""
    try:
        user_id = extract_user_id_from_request(request)
        json_file_path = get_eval_job_path(job_id, user_id)
        
        if not os.path.exists(json_file_path):
            raise HTTPException(status_code=404, detail="Evaluation job not found")
        
        os.remove(json_file_path)
        
        # 메모리에서도 제거
        with jobs_lock:
            if job_id in evaluation_jobs:
                del evaluation_jobs[job_id]
        
        logger.info(f"Deleted evaluation job: {job_id}")
        
        return {
            "job_id": job_id,
            "message": "Successfully deleted evaluation job",
            "status": "success"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting evaluation job {job_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@router.delete("")
async def delete_multiple_evaluation_jobs(
    request: Request,
    job_ids: List[str] = Query(..., description="List of job IDs to delete")
):
    """여러 평가 작업을 한번에 삭제합니다."""
    try:
        user_id = extract_user_id_from_request(request)
        deleted_jobs = []
        failed_jobs = []
        
        for job_id in job_ids:
            try:
                json_file_path = get_eval_job_path(job_id, user_id)
                
                if os.path.exists(json_file_path):
                    os.remove(json_file_path)
                    
                    # 메모리에서 제거
                    with jobs_lock:
                        if job_id in evaluation_jobs:
                            del evaluation_jobs[job_id]
                    
                    deleted_jobs.append(job_id)
                    logger.info(f"Successfully deleted job {job_id}")
                else:
                    failed_jobs.append({
                        "job_id": job_id,
                        "error": "Job not found"
                    })
            
            except Exception as e:
                failed_jobs.append({
                    "job_id": job_id,
                    "error": str(e)
                })
                logger.error(f"Failed to delete job {job_id}: {e}")
        
        return {
            "message": f"Processed {len(job_ids)} jobs",
            "deleted_jobs": deleted_jobs,
            "failed_jobs": failed_jobs,
            "status": "success"
        }
    
    except Exception as e:
        logger.error(f"Error in bulk delete operation: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
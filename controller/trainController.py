"""
Train 컨트롤러

훈련 작업 관리를 위한 중개 API 엔드포인트를 제공합니다.
실제 훈련은 다른 노드의 API를 호출하여 수행합니다.
"""

from fastapi import APIRouter, HTTPException, Request, Query, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import logging
import json
import urllib.request
import urllib.parse
import urllib.error
import os
from controller.controller_helper import extract_user_id_from_request
from datetime import datetime
from zoneinfo import ZoneInfo
from service.database.models.train import TrainMeta
from controller.vastController import CreateInstanceRequest, get_vast_service, _broadcast_status_change
from controller.singletonHelper import get_config_composer, get_vector_manager, get_rag_service, get_document_processor, get_db_manager

logger = logging.getLogger("train-controller")
router = APIRouter(prefix="/api/train", tags=["training"])

# 환경변수에서 타임존 가져오기 (기본값: 서울 시간)
TIMEZONE = ZoneInfo(os.getenv('TIMEZONE', 'Asia/Seoul'))

# ========== Request Models ==========
class MLFlowParams(BaseModel):
    """MLflow 파라미터 요청"""
    mlflow_url: str = Field("https://polar-mlflow-git.x2bee.com/", description="MLFlow URL")
    mlflow_exp_id: str = Field("test", description="MLFlow Experiment ID")
    mlflow_run_id: str = Field("test", description="MLFlow Run ID")

class TrainingStartRequest(BaseModel):
    """훈련 시작 요청"""
    # Common settings
    number_gpu: int = Field(1, description="GPU 개수")
    project_name: str = Field("test-project", description="프로젝트 이름")
    training_method: str = Field("cls", description="훈련 방법")
    model_load_method: str = Field("huggingface", description="모델 로드 방법")
    dataset_load_method: str = Field("huggingface", description="데이터셋 로드 방법")
    hugging_face_user_id: str = Field("CocoRoF", description="HuggingFace 사용자 ID")
    hugging_face_token: str = Field("", description="HuggingFace 토큰")
    mlflow_url: str = Field("https://polar-mlflow-git.x2bee.com/", description="MLflow URL")
    mlflow_run_id: str = Field("test", description="MLflow 실행 ID")
    minio_url: str = Field("polar-store-api.x2bee.com", description="MinIO URL")
    minio_access_key: str = Field("", description="MinIO 액세스 키")
    minio_secret_key: str = Field("", description="MinIO 시크릿 키")

    # DeepSpeed settings
    use_deepspeed: bool = Field(False, description="DeepSpeed 사용 여부")
    ds_jsonpath: str = Field("", description="DeepSpeed JSON 경로")
    ds_preset: str = Field("zero-2", description="DeepSpeed 프리셋")
    ds_stage2_bucket_size: float = Field(5e8, description="DeepSpeed Stage2 버킷 크기")
    ds_stage3_sub_group_size: float = Field(1e9, description="DeepSpeed Stage3 서브그룹 크기")
    ds_stage3_max_live_parameters: float = Field(1e6, description="DeepSpeed Stage3 최대 라이브 파라미터")
    ds_stage3_max_reuse_distance: float = Field(1e6, description="DeepSpeed Stage3 최대 재사용 거리")

    # Model settings
    model_name_or_path: str = Field(..., description="모델 이름 또는 경로")
    language_model_class: str = Field("none", description="언어 모델 클래스")
    ref_model_path: str = Field("", description="참조 모델 경로")
    model_subfolder: str = Field("", description="모델 서브폴더")
    config_name: str = Field("", description="설정 이름")
    tokenizer_name: str = Field("", description="토크나이저 이름")
    cache_dir: str = Field("", description="캐시 디렉토리")

    # Data settings
    train_data: str = Field(..., description="훈련 데이터")
    train_data_dir: str = Field("", description="훈련 데이터 디렉토리")
    train_data_split: str = Field("train", description="훈련 데이터 분할")
    test_data: str = Field("", description="테스트 데이터")
    test_data_dir: str = Field("", description="테스트 데이터 디렉토리")
    test_data_split: str = Field("test", description="테스트 데이터 분할")

    # Dataset column settings
    dataset_main_column: str = Field("instruction", description="데이터셋 메인 컬럼")
    dataset_sub_column: str = Field("output", description="데이터셋 서브 컬럼")
    dataset_minor_column: str = Field("", description="데이터셋 마이너 컬럼")
    dataset_last_column: str = Field("", description="데이터셋 라스트 컬럼")

    # Push settings
    push_to_hub: bool = Field(True, description="허브에 푸시 여부")
    push_to_minio: bool = Field(True, description="MinIO에 푸시 여부")
    minio_model_load_bucket: str = Field("models", description="MinIO 모델 로드 버킷")
    minio_model_save_bucket: str = Field("models", description="MinIO 모델 저장 버킷")
    minio_data_load_bucket: str = Field("data", description="MinIO 데이터 로드 버킷")

    # Training specific settings
    use_sfttrainer: bool = Field(False, description="SFT 트레이너 사용 여부")
    use_dpotrainer: bool = Field(False, description="DPO 트레이너 사용 여부")
    use_ppotrainer: bool = Field(False, description="PPO 트레이너 사용 여부")
    use_grpotrainer: bool = Field(False, description="GRPO 트레이너 사용 여부")
    use_custom_kl_sfttrainer: bool = Field(False, description="커스텀 KL SFT 트레이너 사용 여부")
    mlm_probability: float = Field(0.2, description="MLM 확률")
    num_labels: int = Field(17, description="레이블 수")

    # DPO Setting
    dpo_loss_type: str = Field("sigmoid", description="DPO 손실 타입")
    dpo_beta: float = Field(0.1, description="DPO 베타")
    dpo_label_smoothing: float = Field(0.0, description="DPO 레이블 스무딩")

    # Sentence transformer settings
    st_pooling_mode: str = Field("mean", description="ST 풀링 모드")
    st_dense_feature: int = Field(0, description="ST 덴스 피처")
    st_loss_func: str = Field("CosineSimilarityLoss", description="ST 손실 함수")
    st_evaluation: str = Field("", description="ST 평가")
    st_guide_model: str = Field("nlpai-lab/KURE-v1", description="ST 가이드 모델")
    st_cache_minibatch: int = Field(16, description="ST 캐시 미니배치")
    st_triplet_margin: int = Field(5, description="ST 트리플릿 마진")
    st_cache_gist_temperature: float = Field(0.01, description="ST 캐시 지스트 온도")
    st_use_adaptivelayerloss: bool = Field(False, description="ST 적응형 레이어 손실 사용 여부")
    st_adaptivelayerloss_n_layer: int = Field(4, description="ST 적응형 레이어 손실 레이어 수")

    # Other settings
    use_attn_implementation: bool = Field(True, description="어텐션 구현 사용 여부")
    attn_implementation: str = Field("eager", description="어텐션 구현")
    is_resume: bool = Field(False, description="재개 여부")
    model_commit_msg: str = Field("large-try", description="모델 커밋 메시지")
    train_test_split_ratio: float = Field(0.05, description="훈련/테스트 분할 비율")
    data_filtering: bool = Field(True, description="데이터 필터링 여부")
    tokenizer_max_len: int = Field(256, description="토크나이저 최대 길이")
    output_dir: str = Field("", description="출력 디렉토리")
    overwrite_output_dir: bool = Field(True, description="출력 디렉토리 덮어쓰기 여부")

    # Optimizer settings
    use_stableadamw: bool = Field(True, description="Stable AdamW 사용 여부")
    optim: str = Field("adamw_torch", description="옵티마이저")
    adam_beta1: float = Field(0.900, description="Adam 베타1")
    adam_beta2: float = Field(0.990, description="Adam 베타2")
    adam_epsilon: float = Field(1e-7, description="Adam 엡실론")

    # Saving and evaluation settings
    save_strategy: str = Field("steps", description="저장 전략")
    save_steps: int = Field(1000, description="저장 스텝")
    eval_strategy: str = Field("steps", description="평가 전략")
    eval_steps: int = Field(1000, description="평가 스텝")
    save_total_limit: int = Field(1, description="저장 총 제한")
    hub_model_id: str = Field("", description="허브 모델 ID")
    hub_strategy: str = Field("checkpoint", description="허브 전략")

    # Logging and training settings
    logging_steps: int = Field(5, description="로깅 스텝")
    max_grad_norm: int = Field(1, description="최대 그래디언트 노름")
    per_device_train_batch_size: int = Field(4, description="디바이스당 훈련 배치 크기")
    per_device_eval_batch_size: int = Field(4, description="디바이스당 평가 배치 크기")
    gradient_accumulation_steps: int = Field(16, description="그래디언트 누적 스텝")
    ddp_find_unused_parameters: bool = Field(True, description="DDP 미사용 파라미터 찾기")
    learning_rate: float = Field(2e-5, description="학습률")
    gradient_checkpointing: bool = Field(True, description="그래디언트 체크포인팅")
    num_train_epochs: int = Field(1, description="훈련 에포크 수")
    warmup_ratio: float = Field(0.1, description="워밍업 비율")
    weight_decay: float = Field(0.01, description="가중치 감소")
    do_train: bool = Field(True, description="훈련 수행 여부")
    do_eval: bool = Field(True, description="평가 수행 여부")
    bf16: bool = Field(True, description="BF16 사용 여부")
    fp16: bool = Field(False, description="FP16 사용 여부")

    # PEFT settings
    use_peft: bool = Field(False, description="PEFT 사용 여부")
    peft_type: str = Field("lora", description="PEFT 타입")

    # For LoRA
    lora_target_modules: str = Field("", description="LoRA 타겟 모듈")
    lora_r: int = Field(8, description="LoRA R")
    lora_alpha: int = Field(16, description="LoRA 알파")
    lora_dropout: float = Field(0.05, description="LoRA 드롭아웃")
    lora_modules_to_save: str = Field("", description="LoRA 저장할 모듈")

    # For AdaLoRA
    adalora_init_r: int = Field(12, description="AdaLoRA 초기 R")
    adalora_target_r: int = Field(4, description="AdaLoRA 타겟 R")
    adalora_tinit: int = Field(50, description="AdaLoRA T 초기")
    adalora_tfinal: int = Field(100, description="AdaLoRA T 최종")
    adalora_delta_t: int = Field(10, description="AdaLoRA 델타 T")
    adalora_orth_reg_weight: float = Field(0.5, description="AdaLoRA 직교 정규화 가중치")

    # For IA3
    ia3_target_modules: str = Field("", description="IA3 타겟 모듈")
    feedforward_modules: str = Field("", description="피드포워드 모듈")

    # For LlamaAdapter
    adapter_layers: int = Field(30, description="어댑터 레이어")
    adapter_len: int = Field(16, description="어댑터 길이")

    # For Vera
    vera_target_modules: str = Field("", description="Vera 타겟 모듈")

    # For LayerNorm Tuning
    ln_target_modules: str = Field("", description="LayerNorm 타겟 모듈")

# ========== Helper Functions ==========
def get_train_node_config(request: Request):
    """훈련 노드 설정 가져오기"""
    # user_id = extract_user_id_from_request(request)
    config_composer = get_config_composer(request)

    trainer_host = config_composer.get_config_by_name("TRAINER_HOST").value
    trainer_port = config_composer.get_config_by_name("TRAINER_PORT").value

    if trainer_port:
        return {
            "base_url": f"http://{trainer_host}:{trainer_port}",
            "timeout": 30
        }
    else:
        return {
            "base_url": f"https://{trainer_host}",
            "timeout": 30
        }

def make_external_api_call(url: str, method: str = "GET", data: Dict[str, Any] = None, timeout: int = 30):
    """외부 API 호출 헬퍼 함수"""
    try:
        if data:
            data_encoded = json.dumps(data).encode('utf-8')
            req = urllib.request.Request(
                url,
                data=data_encoded,
                headers={'Content-Type': 'application/json'},
                method=method
            )
        else:
            req = urllib.request.Request(url, method=method)

        response = urllib.request.urlopen(req, timeout=timeout)
        response_data = response.read().decode('utf-8')

        return {
            "success": True,
            "status_code": response.getcode(),
            "data": json.loads(response_data) if response_data else {}
        }
    except urllib.error.HTTPError as e:
        error_data = e.read().decode('utf-8') if e.fp else ""
        return {
            "success": False,
            "status_code": e.code,
            "error": f"HTTP Error {e.code}: {e.reason}",
            "data": error_data
        }
    except Exception as e:
        return {
            "success": False,
            "status_code": 500,
            "error": str(e),
            "data": {}
        }

# ========== API Endpoints ==========

@router.post("/start")
async def start_training(request: Request, training_params: TrainingStartRequest):
    """훈련 작업 시작"""
    try:
        config = get_train_node_config(request)
        url = f"{config['base_url']}/api/train/start"
        user_id = extract_user_id_from_request(request)

        # 요청 파라미터를 딕셔너리로 변환
        params_dict = training_params.dict()
        current_time = datetime.now(TIMEZONE).strftime("%Y%m%d_%H%M%S")
        job_id = f"job_{current_time}"
        params_dict["mlflow_run_id"] = job_id

        # 외부 훈련 노드 API 호출
        result = make_external_api_call(
            url=url,
            method="POST",
            data=params_dict,
            timeout=config['timeout']
        )

        if result["success"]:
            workflow_meta = TrainMeta(
                user_id=user_id,
                model_info_name=params_dict["model_name_or_path"],
                model_info_type=params_dict["model_load_method"],
                train_data=params_dict["train_data"],
                test_data=params_dict["test_data"],
                mlflow_url=params_dict["mlflow_url"],
                mlflow_run_id=params_dict["mlflow_run_id"],
                status="started"
            )
            app_db = get_db_manager(request)
            insert_result = app_db.insert(workflow_meta)

            if insert_result and insert_result.get("result") == "success":
                logger.info(f"Training metadata saved successfully: {workflow_meta}")
            logger.info(f"Training started successfully: {result['data']}")
            return result["data"]
        else:
            logger.error(f"Failed to start training: {result['error']}")
            workflow_meta = TrainMeta(
                user_id=user_id,
                model_info_name=params_dict["model_name_or_path"],
                model_info_type=params_dict["model_load_method"],
                train_data=params_dict["train_data"],
                test_data=params_dict["test_data"],
                mlflow_url=params_dict["mlflow_url"],
                mlflow_run_id=params_dict["mlflow_run_id"],
                status="failed"
            )
            app_db = get_db_manager(request)
            insert_result = app_db.insert(workflow_meta)

            if insert_result and insert_result.get("result") == "success":
                logger.info(f"Training metadata saved successfully: {workflow_meta}")
            raise HTTPException(
                status_code=result["status_code"],
                detail=f"Training start failed: {result['error']}"
            )

    except Exception as e:
        logger.error(f"Error starting training: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/mlflow")
async def get_mlflow(request: Request, params: MLFlowParams):
    """MLflow 정보 조회"""
    try:
        config = get_train_node_config(request)
        url = f"{config['base_url']}/api/train/mlflow"

        # 요청 파라미터를 딕셔너리로 변환
        params_dict = params.dict()

        result = make_external_api_call(
            url=url,
            method="POST",
            data=params_dict,
            timeout=config['timeout']
        )

        if result["success"]:
            return result["data"]
        else:
            raise HTTPException(
                status_code=result["status_code"],
                detail=f"Failed to get MLflow info: {result['error']}"
            )

    except Exception as e:
        logger.error(f"Error getting MLflow info: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/status/{job_id}")
async def get_training_status(request: Request, job_id: str):
    """훈련 작업 상태 조회"""
    try:
        config = get_train_node_config(request)
        url = f"{config['base_url']}/api/train/status/{job_id}"

        result = make_external_api_call(url=url, method="GET", timeout=config['timeout'])

        if result["success"]:
            return result["data"]
        else:
            raise HTTPException(
                status_code=result["status_code"],
                detail=f"Failed to get training status: {result['error']}"
            )

    except Exception as e:
        logger.error(f"Error getting training status: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/jobs")
async def get_all_training_jobs(request: Request):
    """모든 훈련 작업 목록 조회"""
    try:
        config = get_train_node_config(request)
        url = f"{config['base_url']}/api/train/jobs"

        result = make_external_api_call(url=url, method="GET", timeout=config['timeout'])

        if result["success"]:
            return result["data"]
        else:
            raise HTTPException(
                status_code=result["status_code"],
                detail=f"Failed to get training jobs: {result['error']}"
            )

    except Exception as e:
        logger.error(f"Error getting training jobs: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.delete("/stop/{job_id}")
async def stop_training(request: Request, job_id: str):
    """훈련 작업 중지"""
    try:
        config = get_train_node_config(request)
        url = f"{config['base_url']}/api/train/stop/{job_id}"

        result = make_external_api_call(url=url, method="DELETE", timeout=config['timeout'])

        if result["success"]:
            logger.info(f"Training stopped successfully: {job_id}")
            return result["data"]
        else:
            raise HTTPException(
                status_code=result["status_code"],
                detail=f"Failed to stop training: {result['error']}"
            )

    except Exception as e:
        logger.error(f"Error stopping training: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/instances",
    summary="인스턴스 생성",
    description="새로운 VastAI 인스턴스를 생성합니다. 템플릿 사용(budget, high_performance, research) 또는 커스텀 설정 가능합니다.",
    response_model=Dict[str, Any],
    responses={
        400: {"description": "잘못된 요청 (템플릿 없음, 인스턴스 생성 실패 등)"},
        500: {"description": "서버 오류"}
    })
async def create_instance(request: Request, create_request: CreateInstanceRequest, background_tasks: BackgroundTasks):
    try:
        service = get_vast_service(request)

        instance_id = service.create_trainer_instance(
            offer_id=create_request.offer_id,
            template_name=create_request.template_name,
            create_request=create_request
        )

        if not instance_id:
            raise HTTPException(status_code=400, detail="인스턴스 생성 실패")

        is_valid_model = False
        background_tasks.add_task(service.wait_and_setup_instance, instance_id, is_valid_model)

        # 상태 브로드캐스트
        await _broadcast_status_change(instance_id, "creating")

        return {
            "success": True,
            "instance_id": instance_id,
            "template_name": create_request.template_name,
            "message": "인스턴스가 생성되었습니다. 설정이 백그라운드에서 진행됩니다.",
            "status": "creating",
            "tracking_endpoints": {
                "detailed_status": f"/api/vast/instances/{instance_id}",
                "detailed_info": f"/api/vast/instances/{instance_id}/info",
                "update_ports": f"/api/vast/instances/{instance_id}/update-ports"
            },
            "next_steps": [
                f"1. /api/vast/instances/{instance_id} 엔드포인트로 상태를 확인하세요",
                f"2. /api/vast/instances/{instance_id}/info 엔드포인트로 상세 정보를 확인하세요",
                f"3. 포트 매핑이 필요하면 /api/vast/instances/{instance_id}/update-ports를 호출하세요"
            ]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"인스턴스 생성 실패: {e}")
        raise HTTPException(status_code=500, detail="인스턴스 생성 실패")

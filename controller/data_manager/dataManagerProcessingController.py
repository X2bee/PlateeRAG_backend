#/controller/data_manager/dataManagerProcessingController.py
from fastapi import APIRouter, HTTPException, Request, File, UploadFile, Form, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import logging
import os
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
from functools import wraps
import collections
import collections.abc
if not hasattr(collections, 'Callable'):
    collections.Callable = collections.abc.Callable
import mlflow
import json 
import io
import tempfile
from sqlalchemy.exc import OperationalError, DatabaseError
import psycopg2

from controller.helper.singletonHelper import (
    get_db_manager, 
    get_data_manager_registry, 
    get_config_composer
)
from controller.helper.controllerHelper import (
    extract_user_id_from_request, 
    validate_db_config, 
    parse_connection_url, 
    create_db_connection_string,
    parse_db_error
)

router = APIRouter(prefix="/processing", tags=["data-manager"])
logger = logging.getLogger("data-manager-controller")

# ========== Configuration ==========
def get_mlflow_config(config_composer, request_uri: Optional[str] = None) -> Dict[str, str]:
    """MLflow 관련 설정 가져오기"""
    config = {
        'tracking_uri': request_uri or "https://polar-mlflow-git.x2bee.com",
        's3_access_key': 'minioadmin',
        's3_secret_key': 'minioadmin123',
        's3_endpoint': 'https://minio.x2bee.com'
    }
    
    try:
        if not request_uri:
            config['tracking_uri'] = config_composer.get_config_by_name("MLFLOW_TRACKING_URI").value
    except Exception as e:
        logger.warning(f"MLFLOW_TRACKING_URI 설정 가져오기 실패: {e}")
    
    return config

def setup_mlflow_env(config: Dict[str, str]):
    """MLflow 환경 변수 설정"""
    os.environ['AWS_ACCESS_KEY_ID'] = config['s3_access_key']
    os.environ['AWS_SECRET_ACCESS_KEY'] = config['s3_secret_key']
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = config['s3_endpoint']
    mlflow.set_tracking_uri(config['tracking_uri'])

# ========== Request Models ==========
class BaseManagerRequest(BaseModel):
    """기본 매니저 요청 모델"""
    manager_id: str = Field(..., description="매니저 ID")

class DatabaseConnectionRequest(BaseModel):
    """데이터베이스 연결 요청 (공통)"""
    db_config: Optional[Dict[str, Any]] = Field(None, description="데이터베이스 연결 설정")
    connection_url: Optional[str] = Field(None, description="연결 URL")

class LoadFromDatabaseRequest(BaseManagerRequest):
    """데이터베이스 로드 요청"""
    db_config: Optional[Dict[str, Any]] = Field(None, description="데이터베이스 연결 설정")
    connection_url: Optional[str] = Field(None, description="연결 URL")
    query: Optional[str] = Field(None, description="SQL 쿼리")
    table_name: Optional[str] = Field(None, description="테이블명")
    schema: Optional[str] = Field(None, description="스키마명 (PostgreSQL)")
    chunk_size: Optional[int] = Field(None, description="청크 크기", ge=1000)

class TableListRequest(DatabaseConnectionRequest):
    """테이블 목록 조회 요청"""
    schema: Optional[str] = Field(None, description="스키마명")

class TablePreviewRequest(DatabaseConnectionRequest):
    """테이블 미리보기 요청"""
    table_name: str = Field(..., description="테이블명")
    schema: Optional[str] = Field(None, description="스키마명")
    limit: int = Field(10, description="조회할 행 수", ge=1, le=100)

class QueryValidationRequest(DatabaseConnectionRequest):
    """쿼리 검증 요청"""
    query: str = Field(..., description="SQL 쿼리")

class DownloadDatasetRequest(BaseManagerRequest):
    """데이터셋 다운로드 요청"""
    repo_id: str = Field(..., description="Huggingface 리포지토리 ID")
    filename: Optional[str] = Field(None, description="특정 파일명")
    split: Optional[str] = Field(None, description="데이터 분할")

class ExportDatasetRequest(BaseManagerRequest):
    """데이터셋 내보내기 요청"""
    pass

class DropColumnsRequest(BaseManagerRequest):
    """컬럼 삭제 요청"""
    columns: List[str] = Field(..., description="삭제할 컬럼명들", min_items=1)

class ReplaceValuesRequest(BaseManagerRequest):
    """값 교체 요청"""
    column_name: str = Field(..., description="대상 컬럼명")
    old_value: str = Field(..., description="교체할 기존 값")
    new_value: str = Field(..., description="새로운 값")

class ApplyOperationRequest(BaseManagerRequest):
    """연산 적용 요청"""
    column_name: str = Field(..., description="대상 컬럼명")
    operation: str = Field(..., description="연산식", pattern=r'^[+\-*/\d.]+$')

class RemoveNullRowsRequest(BaseManagerRequest):
    """NULL row 제거 요청"""
    column_name: Optional[str] = Field(None, description="특정 컬럼명")

class UploadToHfRequest(BaseManagerRequest):
    """HuggingFace Hub 업로드 요청"""
    repo_id: str = Field(..., description="리포지토리 ID")
    filename: Optional[str] = Field(None, description="파일명")
    private: bool = Field(False, description="프라이빗 여부")
    hf_user_id: Optional[str] = Field(None, description="HF 사용자 ID")
    hub_token: Optional[str] = Field(None, description="HF Hub 토큰")

class CopyColumnRequest(BaseManagerRequest):
    """컬럼 복사 요청"""
    source_column: str = Field(..., description="원본 컬럼명")
    new_column: str = Field(..., description="새 컬럼명")

class RenameColumnRequest(BaseManagerRequest):
    """컬럼 이름 변경 요청"""
    old_name: str = Field(..., description="기존 컬럼명")
    new_name: str = Field(..., description="새 컬럼명")

class FormatColumnsRequest(BaseManagerRequest):
    """컬럼 문자열 포맷팅 요청"""
    column_names: List[str] = Field(..., description="컬럼명들", min_items=1)
    template: str = Field(..., description="문자열 템플릿")
    new_column: str = Field(..., description="새 컬럼명")

class CalculateColumnsRequest(BaseManagerRequest):
    """컬럼 간 사칙연산 요청"""
    col1: str = Field(..., description="첫 번째 컬럼명")
    col2: str = Field(..., description="두 번째 컬럼명")
    operation: str = Field(..., description="연산자", pattern=r'^[+\-*/]$')
    new_column: str = Field(..., description="새 컬럼명")

class ExecuteCallbackRequest(BaseManagerRequest):
    """사용자 콜백 코드 실행 요청"""
    callback_code: str = Field(..., description="PyArrow 코드", min_length=1)

class UploadToMlflowRequest(BaseManagerRequest):
    """MLflow 업로드 요청"""
    experiment_name: str = Field(..., description="실험 이름")
    artifact_path: str = Field("dataset", description="아티팩트 경로")
    dataset_name: str = Field(..., description="데이터셋 이름")
    description: Optional[str] = Field(None, description="설명")
    tags: Optional[Dict[str, str]] = Field(None, description="태그")
    format: str = Field("parquet", description="저장 형식", pattern="^(parquet|csv)$")
    mlflow_tracking_uri: Optional[str] = Field(None, description="MLflow URI")

class ListMlflowDatasetsRequest(BaseModel):
    """MLflow 데이터셋 목록 조회 요청"""
    experiment_name: Optional[str] = Field(None, description="실험명 필터")
    max_results: int = Field(100, description="최대 결과 개수", ge=1, le=1000)
    mlflow_tracking_uri: Optional[str] = Field(None, description="MLflow URI")

class GetMLflowDatasetColumnsRequest(BaseModel):
    """MLflow 데이터셋 컬럼 조회 요청"""
    run_id: str = Field(..., description="Run ID")
    artifact_path: Optional[str] = Field("dataset", description="아티팩트 경로")
    mlflow_tracking_uri: Optional[str] = Field(None, description="MLflow URI")

# ========== Decorators & Helper Functions ==========
def with_auth_manager(func: Callable) -> Callable:
    """매니저 인증 데코레이터"""
    @wraps(func)
    async def wrapper(request: Request, *args, **kwargs):
        user_id = extract_user_id_from_request(request)
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID가 제공되지 않았습니다")
        
        # manager_id 추출 (첫 번째 인자 또는 키워드 인자에서)
        manager_id = None
        if args and hasattr(args[0], 'manager_id'):
            manager_id = args[0].manager_id
        elif 'manager_id' in kwargs:
            manager_id = kwargs['manager_id']
        
        if manager_id:
            registry = get_data_manager_registry(request)
            manager = registry.get_manager(manager_id, user_id)
            if not manager:
                raise HTTPException(
                    status_code=404,
                    detail=f"매니저 '{manager_id}'를 찾을 수 없거나 접근 권한이 없습니다"
                )
            kwargs['manager'] = manager
        
        kwargs['user_id'] = user_id
        kwargs['request'] = request
        return await func(*args, **kwargs)
    return wrapper

def handle_errors(operation_name: str):
    """에러 처리 데코레이터"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except HTTPException:
                raise
            except RuntimeError as e:
                logger.error(f"{operation_name} 실패: {e}")
                raise HTTPException(status_code=500, detail=str(e))
            except Exception as e:
                logger.error(f"{operation_name} 중 예상치 못한 오류: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"{operation_name} 중 오류가 발생했습니다")
        return wrapper
    return decorator

def get_db_config_from_request(db_request: DatabaseConnectionRequest) -> Dict[str, Any]:
    """요청에서 DB 설정 추출 (URL 우선)"""
    if db_request.connection_url:
        try:
            return parse_connection_url(db_request.connection_url)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    if not db_request.db_config:
        raise HTTPException(status_code=400, detail="db_config 또는 connection_url 중 하나가 필요합니다")
    
    validate_db_config(db_request.db_config)
    return db_request.db_config

def create_response(
    success: bool,
    message: str,
    manager_id: Optional[str] = None,
    **additional_data
) -> Dict[str, Any]:
    """표준 응답 생성"""
    response = {
        "success": success,
        "message": message
    }
    if manager_id:
        response["manager_id"] = manager_id
    response.update(additional_data)
    return response

# ========== API Endpoints - HuggingFace ==========
@router.post("/hf/download-dataset",
    summary="HF 데이터셋 다운로드",
    response_model=Dict[str, Any])
@handle_errors("데이터셋 다운로드")
async def hf_download_dataset(
    request: Request, 
    download_request: DownloadDatasetRequest
) -> Dict[str, Any]:
    """HuggingFace에서 데이터셋 다운로드 및 적재"""
    user_id = extract_user_id_from_request(request)
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID가 제공되지 않았습니다")

    registry = get_data_manager_registry(request)
    manager = registry.get_manager(download_request.manager_id, user_id)
    if not manager:
        raise HTTPException(status_code=404, detail="매니저를 찾을 수 없습니다")

    result = manager.hf_download_and_load_dataset(
        repo_id=download_request.repo_id,
        filename=download_request.filename,
        split=download_request.split
    )

    log_msg = f"Dataset {'re-loaded' if result.get('is_new_version') else 'loaded'} (v{result['load_count']})"
    logger.info(f"{log_msg}: {download_request.manager_id} from {download_request.repo_id}")

    return create_response(
        success=True,
        message=f"데이터셋이 성공적으로 다운로드되고 적재되었습니다 (v{result['load_count']})",
        manager_id=download_request.manager_id,
        download_info=result,
        version_info={
            "dataset_id": result.get("dataset_id"),
            "load_count": result.get("load_count"),
            "is_new_version": result.get("is_new_version", False)
        }
    )

# ========== API Endpoints - Local Upload ==========
@router.post("/local/upload-dataset",
    summary="로컬 파일 업로드",
    response_model=Dict[str, Any])
@handle_errors("로컬 데이터셋 업로드")
async def local_upload_dataset(
    request: Request,
    files: List[UploadFile] = File(...),
    manager_id: str = Form(...)
) -> Dict[str, Any]:
    """로컬 파일 업로드 및 적재"""
    user_id = extract_user_id_from_request(request)
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID가 제공되지 않았습니다")

    registry = get_data_manager_registry(request)
    manager = registry.get_manager(manager_id, user_id)
    if not manager:
        raise HTTPException(status_code=404, detail="매니저를 찾을 수 없습니다")

    # 파일 형식 검증
    file_extensions = [file.filename.split('.')[-1].lower() for file in files]
    unique_extensions = set(file_extensions)

    if len(unique_extensions) > 1:
        raise HTTPException(
            status_code=400,
            detail=f"모든 파일이 같은 형식이어야 합니다. 업로드된 형식: {list(unique_extensions)}"
        )

    file_type = file_extensions[0]
    if file_type not in ['parquet', 'csv']:
        raise HTTPException(
            status_code=400,
            detail=f"지원되지 않는 파일 형식: {file_type}"
        )

    uploaded_files = [file.file for file in files]
    filenames = [file.filename for file in files]
    result = manager.local_upload_and_load_dataset(uploaded_files, filenames)

    log_msg = f"로컬 데이터셋 {'재업로드' if result.get('is_new_version') else '초기 업로드'}"
    logger.info(f"{log_msg} (v{result['load_count']}): {manager_id}, {len(files)}개 파일")

    return create_response(
        success=True,
        message=f"{len(files)}개 파일이 성공적으로 업로드되고 적재되었습니다 (v{result['load_count']})",
        manager_id=manager_id,
        dataset_info=result,
        version_info={
            "dataset_id": result.get("dataset_id"),
            "load_count": result.get("load_count"),
            "is_new_version": result.get("is_new_version", False)
        }
    )

# ========== API Endpoints - Export ==========
@router.post("/export/csv", summary="CSV 내보내기")
@handle_errors("CSV 내보내기")
async def export_dataset_as_csv(request: Request, export_request: ExportDatasetRequest):
    """데이터셋을 CSV 파일로 다운로드"""
    user_id = extract_user_id_from_request(request)
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID가 제공되지 않았습니다")

    registry = get_data_manager_registry(request)
    manager = registry.get_manager(export_request.manager_id, user_id)
    if not manager:
        raise HTTPException(status_code=404, detail="매니저를 찾을 수 없습니다")

    file_path = manager.download_dataset_as_csv()
    filename = f"dataset_{export_request.manager_id}.csv"

    logger.info(f"CSV 다운로드 요청: {export_request.manager_id}")

    return FileResponse(
        path=file_path,
        filename=filename,
        media_type='text/csv',
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

@router.post("/export/parquet", summary="Parquet 내보내기")
@handle_errors("Parquet 내보내기")
async def export_dataset_as_parquet(request: Request, export_request: ExportDatasetRequest):
    """데이터셋을 Parquet 파일로 다운로드"""
    user_id = extract_user_id_from_request(request)
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID가 제공되지 않았습니다")

    registry = get_data_manager_registry(request)
    manager = registry.get_manager(export_request.manager_id, user_id)
    if not manager:
        raise HTTPException(status_code=404, detail="매니저를 찾을 수 없습니다")

    file_path = manager.download_dataset_as_parquet()
    filename = f"dataset_{export_request.manager_id}.parquet"

    logger.info(f"Parquet 다운로드 요청: {export_request.manager_id}")

    return FileResponse(
        path=file_path,
        filename=filename,
        media_type='application/octet-stream',
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

# ========== API Endpoints - Statistics & Operations ==========
@router.post("/statistics", summary="기술통계정보 조회", response_model=Dict[str, Any])
@handle_errors("통계정보 생성")
async def get_dataset_statistics(request: Request, export_request: ExportDatasetRequest) -> Dict[str, Any]:
    """데이터셋 기술통계정보 조회"""
    user_id = extract_user_id_from_request(request)
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID가 제공되지 않았습니다")

    registry = get_data_manager_registry(request)
    manager = registry.get_manager(export_request.manager_id, user_id)
    if not manager:
        raise HTTPException(status_code=404, detail="매니저를 찾을 수 없습니다")

    statistics = manager.get_dataset_statistics()
    logger.info(f"통계정보 조회: {export_request.manager_id}")

    return create_response(
        success=True,
        message="데이터셋 기술통계정보가 성공적으로 생성되었습니다",
        manager_id=export_request.manager_id,
        statistics=statistics
    )

@router.post("/drop-columns", summary="컬럼 삭제", response_model=Dict[str, Any])
@handle_errors("컬럼 삭제")
async def drop_dataset_columns(request: Request, drop_request: DropColumnsRequest) -> Dict[str, Any]:
    """데이터셋 컬럼 삭제"""
    user_id = extract_user_id_from_request(request)
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID가 제공되지 않았습니다")

    registry = get_data_manager_registry(request)
    manager = registry.get_manager(drop_request.manager_id, user_id)
    if not manager:
        raise HTTPException(status_code=404, detail="매니저를 찾을 수 없습니다")

    result_info = manager.drop_dataset_columns(drop_request.columns)
    logger.info(f"컬럼 삭제: {drop_request.manager_id}, {drop_request.columns}")

    return create_response(
        success=True,
        message=f"{len(drop_request.columns)}개 컬럼이 성공적으로 삭제되었습니다",
        manager_id=drop_request.manager_id,
        drop_info=result_info
    )

@router.post("/replace-values", summary="컬럼 값 교체", response_model=Dict[str, Any])
@handle_errors("값 교체")
async def replace_column_values(request: Request, replace_request: ReplaceValuesRequest) -> Dict[str, Any]:
    """컬럼 값 교체"""
    user_id = extract_user_id_from_request(request)
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID가 제공되지 않았습니다")

    registry = get_data_manager_registry(request)
    manager = registry.get_manager(replace_request.manager_id, user_id)
    if not manager:
        raise HTTPException(status_code=404, detail="매니저를 찾을 수 없습니다")

    result_info = manager.replace_dataset_column_values(
        replace_request.column_name,
        replace_request.old_value,
        replace_request.new_value
    )

    logger.info(f"값 교체: {replace_request.manager_id}, {replace_request.column_name}")

    return create_response(
        success=True,
        message=f"컬럼 '{replace_request.column_name}'에서 값이 성공적으로 교체되었습니다",
        manager_id=replace_request.manager_id,
        replace_info=result_info
    )

@router.post("/apply-operation", summary="컬럼 연산 적용", response_model=Dict[str, Any])
@handle_errors("연산 적용")
async def apply_column_operation(request: Request, operation_request: ApplyOperationRequest) -> Dict[str, Any]:
    """컬럼 연산 적용"""
    user_id = extract_user_id_from_request(request)
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID가 제공되지 않았습니다")

    registry = get_data_manager_registry(request)
    manager = registry.get_manager(operation_request.manager_id, user_id)
    if not manager:
        raise HTTPException(status_code=404, detail="매니저를 찾을 수 없습니다")

    result_info = manager.apply_dataset_column_operation(
        operation_request.column_name,
        operation_request.operation
    )

    logger.info(f"연산 적용: {operation_request.manager_id}, {operation_request.column_name}, {operation_request.operation}")

    return create_response(
        success=True,
        message=f"컬럼 '{operation_request.column_name}'에 연산이 성공적으로 적용되었습니다",
        manager_id=operation_request.manager_id,
        operation_info=result_info
    )

@router.post("/remove-null-rows", summary="NULL row 제거", response_model=Dict[str, Any])
@handle_errors("NULL row 제거")
async def remove_null_rows(request: Request, remove_request: RemoveNullRowsRequest) -> Dict[str, Any]:
    """NULL 값이 있는 행 제거"""
    user_id = extract_user_id_from_request(request)
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID가 제공되지 않았습니다")

    registry = get_data_manager_registry(request)
    manager = registry.get_manager(remove_request.manager_id, user_id)
    if not manager:
        raise HTTPException(status_code=404, detail="매니저를 찾을 수 없습니다")

    result_info = manager.remove_null_rows_from_dataset(remove_request.column_name)

    response_msg = (
        f"컬럼 '{remove_request.column_name}'에서 NULL 값이 있는 행이 제거되었습니다"
        if remove_request.column_name
        else "NULL 값이 있는 행이 제거되었습니다"
    )

    logger.info(f"NULL row 제거: {remove_request.manager_id}, {result_info['removed_rows']}개 행")

    return create_response(
        success=True,
        message=response_msg,
        manager_id=remove_request.manager_id,
        removal_info=result_info
    )

@router.post("/copy-column", summary="컬럼 복사", response_model=Dict[str, Any])
@handle_errors("컬럼 복사")
async def copy_dataset_column(request: Request, copy_request: CopyColumnRequest) -> Dict[str, Any]:
    """컬럼 복사"""
    user_id = extract_user_id_from_request(request)
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID가 제공되지 않았습니다")

    registry = get_data_manager_registry(request)
    manager = registry.get_manager(copy_request.manager_id, user_id)
    if not manager:
        raise HTTPException(status_code=404, detail="매니저를 찾을 수 없습니다")

    result_info = manager.copy_dataset_column(
        copy_request.source_column,
        copy_request.new_column
    )

    logger.info(f"컬럼 복사: {copy_request.manager_id}, {copy_request.source_column} → {copy_request.new_column}")

    return create_response(
        success=True,
        message=f"컬럼 '{copy_request.source_column}'이 '{copy_request.new_column}'으로 복사되었습니다",
        manager_id=copy_request.manager_id,
        copy_info=result_info
    )

@router.post("/rename-column", summary="컬럼 이름 변경", response_model=Dict[str, Any])
@handle_errors("컬럼 이름 변경")
async def rename_dataset_column(request: Request, rename_request: RenameColumnRequest) -> Dict[str, Any]:
    """컬럼 이름 변경"""
    user_id = extract_user_id_from_request(request)
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID가 제공되지 않았습니다")

    registry = get_data_manager_registry(request)
    manager = registry.get_manager(rename_request.manager_id, user_id)
    if not manager:
        raise HTTPException(status_code=404, detail="매니저를 찾을 수 없습니다")

    result_info = manager.rename_dataset_column(
        rename_request.old_name,
        rename_request.new_name
    )

    logger.info(f"컬럼 이름 변경: {rename_request.manager_id}, {rename_request.old_name} → {rename_request.new_name}")

    return create_response(
        success=True,
        message=f"컬럼 이름이 '{rename_request.old_name}'에서 '{rename_request.new_name}'으로 변경되었습니다",
        manager_id=rename_request.manager_id,
        rename_info=result_info
    )

@router.post("/format-columns", summary="컬럼 문자열 포맷팅", response_model=Dict[str, Any])
@handle_errors("컬럼 문자열 포맷팅")
async def format_dataset_columns(request: Request, format_request: FormatColumnsRequest) -> Dict[str, Any]:
    """컬럼 문자열 포맷팅"""
    user_id = extract_user_id_from_request(request)
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID가 제공되지 않았습니다")

    registry = get_data_manager_registry(request)
    manager = registry.get_manager(format_request.manager_id, user_id)
    if not manager:
        raise HTTPException(status_code=404, detail="매니저를 찾을 수 없습니다")

    result_info = manager.format_columns_to_string(
        format_request.column_names,
        format_request.template,
        format_request.new_column
    )

    logger.info(f"컬럼 포맷팅: {format_request.manager_id}, {format_request.column_names} → {format_request.new_column}")

    return create_response(
        success=True,
        message=f"새로운 컬럼 '{format_request.new_column}'이 생성되었습니다",
        manager_id=format_request.manager_id,
        format_info=result_info
    )

@router.post("/calculate-columns", summary="컬럼 간 사칙연산", response_model=Dict[str, Any])
@handle_errors("컬럼 간 연산")
async def calculate_dataset_columns(request: Request, calc_request: CalculateColumnsRequest) -> Dict[str, Any]:
    """컬럼 간 사칙연산"""
    user_id = extract_user_id_from_request(request)
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID가 제공되지 않았습니다")

    registry = get_data_manager_registry(request)
    manager = registry.get_manager(calc_request.manager_id, user_id)
    if not manager:
        raise HTTPException(status_code=404, detail="매니저를 찾을 수 없습니다")

    result_info = manager.calculate_columns_to_new(
        calc_request.col1,
        calc_request.col2,
        calc_request.operation,
        calc_request.new_column
    )

    logger.info(f"컬럼 연산: {calc_request.manager_id}, {calc_request.col1} {calc_request.operation} {calc_request.col2}")

    return create_response(
        success=True,
        message=f"연산 결과가 '{calc_request.new_column}'에 저장되었습니다",
        manager_id=calc_request.manager_id,
        calculation_info=result_info
    )

@router.post("/execute-callback", summary="사용자 콜백 실행", response_model=Dict[str, Any])
@handle_errors("사용자 콜백 실행")
async def execute_dataset_callback(request: Request, callback_request: ExecuteCallbackRequest) -> Dict[str, Any]:
    """사용자 콜백 코드 실행"""
    user_id = extract_user_id_from_request(request)
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID가 제공되지 않았습니다")

    registry = get_data_manager_registry(request)
    manager = registry.get_manager(callback_request.manager_id, user_id)
    if not manager:
        raise HTTPException(status_code=404, detail="매니저를 찾을 수 없습니다")

    result_info = manager.execute_dataset_callback(callback_request.callback_code)

    logger.info(f"콜백 실행: {callback_request.manager_id}, {result_info['rows_changed']:+d}행, {result_info['columns_changed']:+d}열")

    return create_response(
        success=True,
        message=f"콜백 실행 완료: {result_info['rows_changed']:+d}행, {result_info['columns_changed']:+d}열 변경",
        manager_id=callback_request.manager_id,
        callback_result=result_info
    )

# ========== API Endpoints - HuggingFace Upload ==========
@router.post("/upload-to-hf", summary="HuggingFace Hub 업로드", response_model=Dict[str, Any])
@handle_errors("HuggingFace 업로드")
async def upload_dataset_to_hf(request: Request, upload_request: UploadToHfRequest) -> Dict[str, Any]:
    """데이터셋을 HuggingFace Hub에 업로드"""
    user_id = extract_user_id_from_request(request)
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID가 제공되지 않았습니다")

    registry = get_data_manager_registry(request)
    manager = registry.get_manager(upload_request.manager_id, user_id)
    if not manager:
        raise HTTPException(status_code=404, detail="매니저를 찾을 수 없습니다")

    config_composer = get_config_composer(request)

    # HuggingFace 설정값 결정
    hf_user_id = upload_request.hf_user_id
    hub_token = upload_request.hub_token

    if not hf_user_id:
        try:
            hf_user_id = config_composer.get_config_by_name("HUGGING_FACE_USER_ID").value
        except Exception:
            raise HTTPException(status_code=400, detail="HuggingFace User ID가 필요합니다")

    if not hub_token:
        try:
            hub_token = config_composer.get_config_by_name("HUGGING_FACE_HUB_TOKEN").value
        except Exception:
            raise HTTPException(status_code=400, detail="HuggingFace Hub Token이 필요합니다")

    result_info = manager.upload_dataset_to_hf_repo(
        repo_id=upload_request.repo_id,
        hf_user_id=hf_user_id,
        hub_token=hub_token,
        filename=upload_request.filename,
        private=upload_request.private
    )

    logger.info(f"HF 업로드 완료: {upload_request.manager_id} → {result_info['repo_id']}")

    return create_response(
        success=True,
        message="데이터셋이 HuggingFace Hub에 업로드되었습니다",
        manager_id=upload_request.manager_id,
        upload_info=result_info
    )

# ========== API Endpoints - MLflow ==========
@router.post("/upload-to-mlflow", summary="MLflow 업로드", response_model=Dict[str, Any])
@handle_errors("MLflow 업로드")
async def upload_dataset_to_mlflow(request: Request, upload_request: UploadToMlflowRequest) -> Dict[str, Any]:
    """MLflow에 데이터셋 업로드 (계보 정보 포함)"""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise HTTPException(status_code=500, detail="필수 패키지가 설치되지 않았습니다")

    user_id = extract_user_id_from_request(request)
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID가 제공되지 않았습니다")

    registry = get_data_manager_registry(request)
    manager = registry.get_manager(upload_request.manager_id, user_id)
    if not manager:
        raise HTTPException(status_code=404, detail="매니저를 찾을 수 없습니다")

    # MLflow 설정
    config_composer = get_config_composer(request)
    mlflow_config = get_mlflow_config(config_composer, upload_request.mlflow_tracking_uri)
    setup_mlflow_env(mlflow_config)
    
    mlflow.set_experiment(upload_request.experiment_name)
    
    with mlflow.start_run() as run:
        logger.info(f"MLflow Run 시작: {run.info.run_id}, experiment={upload_request.experiment_name}")
        
        # 1. 데이터셋 파일 생성 및 업로드
        if upload_request.format == "parquet":
            file_path = manager.download_dataset_as_parquet()
        else:
            file_path = manager.download_dataset_as_csv()
        
        mlflow.log_artifact(file_path, artifact_path=upload_request.artifact_path)
        logger.info(f"데이터셋 파일 업로드 완료: {os.path.basename(file_path)}")
        
        # 2. 계보(Lineage) 정보 생성
        version_history = manager.get_version_history()
        source_info = manager.redis_manager.get_source_info(manager.manager_id) if manager.redis_manager else {}
        
        lineage_info = {
            "manager_id": upload_request.manager_id,
            "dataset_id": manager.dataset_id,
            "original_source": source_info,
            "current_version": manager.current_version - 1,
            "total_operations": len(version_history),
            "version_history": version_history,
            "transformations": [v["operation"] for v in version_history],
            "final_checksum": manager._calculate_checksum(manager.dataset),
            "uploaded_at": datetime.now().isoformat(),
            "uploaded_by": user_id
        }
        
        # 3. Lineage JSON 저장
        with tempfile.NamedTemporaryFile(mode='w', suffix='_lineage.json', delete=False) as f:
            json.dump(lineage_info, f, indent=2, default=str)
            lineage_path = f.name
        
        mlflow.log_artifact(lineage_path, artifact_path=upload_request.artifact_path)
        
        # 4. 통계 정보 저장
        statistics = manager.get_dataset_statistics()
        with tempfile.NamedTemporaryFile(mode='w', suffix='_stats.json', delete=False) as f:
            json.dump(statistics, f, indent=2, default=str)
            stats_path = f.name
        
        mlflow.log_artifact(stats_path, artifact_path=upload_request.artifact_path)
        
        # 5. 메트릭 로깅
        if isinstance(statistics.get('statistics'), dict):
            dataset_info = statistics['statistics'].get('dataset_info', {})
            mlflow.log_metric("dataset_rows", dataset_info.get('total_rows', 0))
            mlflow.log_metric("dataset_columns", dataset_info.get('total_columns', 0))
            mlflow.log_metric("total_versions", len(version_history))
        
        # 6. 파라미터 로깅
        params = {
            "dataset_name": upload_request.dataset_name,
            "format": upload_request.format,
            "manager_id": upload_request.manager_id,
            "dataset_id": manager.dataset_id,
            "current_version": manager.current_version - 1
        }
        
        if source_info:
            params["source_type"] = source_info.get("type")
            if source_info.get("type") == "huggingface":
                params["source_repo"] = source_info.get("repo_id")
            params["original_checksum"] = source_info.get("checksum")
        
        mlflow.log_params(params)
        
        # 7. 태그 설정
        tags = {
            "dataset_name": upload_request.dataset_name,
            "format": upload_request.format,
            "user_id": user_id,
            "manager_id": upload_request.manager_id,
            "dataset_id": manager.dataset_id,
            "final_checksum": lineage_info["final_checksum"],
            "source_type": source_info.get("type") if source_info else "unknown"
        }
        
        if upload_request.tags:
            tags.update(upload_request.tags)
        
        mlflow.set_tags(tags)
        
        # 8. Redis에 Run 정보 저장
        if manager.redis_manager:
            manager.redis_manager.add_mlflow_run(
                upload_request.manager_id,
                run.info.run_id,
                {
                    "experiment_name": upload_request.experiment_name,
                    "dataset_name": upload_request.dataset_name,
                    "artifact_uri": run.info.artifact_uri,
                    "version_at_upload": manager.current_version - 1
                }
            )
        
        # 9. MinIO의 processed-datasets에도 저장
        if manager.minio_storage:
            try:
                # Parquet 파일 저장
                buffer = io.BytesIO()
                pq.write_table(manager.dataset, buffer)
                buffer.seek(0)
                
                processed_path = f"{upload_request.experiment_name}/{run.info.run_id}/dataset.parquet"
                manager.minio_storage.client.put_object(
                    "processed-datasets",
                    processed_path,
                    buffer,
                    length=buffer.getbuffer().nbytes,
                    content_type="application/octet-stream"
                )
                
                # Lineage 정보 저장
                lineage_buffer = io.BytesIO(json.dumps(lineage_info, indent=2, default=str).encode())
                lineage_minio_path = f"{upload_request.experiment_name}/{run.info.run_id}/lineage.json"
                manager.minio_storage.client.put_object(
                    "processed-datasets",
                    lineage_minio_path,
                    lineage_buffer,
                    length=lineage_buffer.getbuffer().nbytes,
                    content_type="application/json"
                )
                
                logger.info(f"MinIO 저장 완료: {processed_path}")
                
            except Exception as e:
                logger.warning(f"MinIO 저장 실패 (MLflow는 성공): {e}")
        
        # 임시 파일 정리
        try:
            os.unlink(file_path)
            os.unlink(lineage_path)
            os.unlink(stats_path)
        except Exception as e:
            logger.warning(f"임시 파일 정리 실패: {e}")
    
    logger.info(f"MLflow 업로드 완료: run_id={run.info.run_id}")
    
    return create_response(
        success=True,
        message="데이터셋이 버전 정보와 함께 MLflow에 업로드되었습니다",
        manager_id=upload_request.manager_id,
        mlflow_info={
            "run_id": run.info.run_id,
            "experiment_name": upload_request.experiment_name,
            "dataset_name": upload_request.dataset_name,
            "artifact_uri": run.info.artifact_uri,
            "version": manager.current_version - 1,
            "total_operations": len(version_history),
            "lineage_saved": True
        }
    )

@router.post("/mlflow/list-datasets", summary="MLflow 데이터셋 목록", response_model=Dict[str, Any])
@handle_errors("MLflow 데이터셋 목록 조회")
async def list_mlflow_datasets(request: Request, list_request: ListMlflowDatasetsRequest) -> Dict[str, Any]:
    """MLflow 업로드된 데이터셋 목록 조회"""
    try:
        import mlflow
    except ImportError:
        raise HTTPException(status_code=500, detail="MLflow 패키지가 설치되지 않았습니다")

    user_id = extract_user_id_from_request(request)
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID가 제공되지 않았습니다")

    # MLflow 설정
    config_composer = get_config_composer(request)
    mlflow_config = get_mlflow_config(config_composer, list_request.mlflow_tracking_uri)
    setup_mlflow_env(mlflow_config)
    
    client = mlflow.tracking.MlflowClient()
    
    logger.info(f"MLflow 데이터셋 목록 조회 시작 (user_id: {user_id})")
    
    # 실험 목록 가져오기
    if list_request.experiment_name:
        try:
            experiment = client.get_experiment_by_name(list_request.experiment_name)
            experiments = [experiment] if experiment else []
        except Exception as e:
            logger.warning(f"실험 '{list_request.experiment_name}' 조회 실패: {e}")
            experiments = []
    else:
        experiments = client.search_experiments(max_results=1000)
    
    datasets = []
    
    for experiment in experiments:
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            max_results=list_request.max_results,
            order_by=["start_time DESC"]
        )
        
        for run in runs:
            tags = run.data.tags
            if "dataset_name" not in tags:
                continue
            
            # user_id 필터링
            if tags.get("user_id") != str(user_id):
                continue
            
            params = run.data.params
            metrics = run.data.metrics
            
            # 아티팩트 목록 조회
            artifact_list = []
            try:
                artifacts = client.list_artifacts(run.info.run_id)
                artifact_list = [
                    {
                        "path": a.path,
                        "size_bytes": a.file_size,
                        "is_dir": a.is_dir
                    }
                    for a in artifacts
                ]
            except Exception as e:
                logger.warning(f"아티팩트 조회 실패 (run {run.info.run_id}): {e}")
            
            dataset_info = {
                "run_id": run.info.run_id,
                "experiment_name": experiment.name,
                "experiment_id": experiment.experiment_id,
                "dataset_name": tags.get("dataset_name"),
                "format": tags.get("format"),
                "manager_id": tags.get("manager_id"),
                "upload_user_id": tags.get("user_id"),
                "created_at": datetime.fromtimestamp(run.info.start_time / 1000).isoformat(),
                "artifact_uri": run.info.artifact_uri,
                "status": run.info.status,
                "metrics": {
                    "rows": metrics.get("dataset_rows"),
                    "columns": metrics.get("dataset_columns"),
                    "size_mb": metrics.get("dataset_size_mb")
                },
                "description": params.get("description"),
                "artifacts": artifact_list,
                "run_url": f"{mlflow_config['tracking_uri']}/#/experiments/{experiment.experiment_id}/runs/{run.info.run_id}"
            }
            
            datasets.append(dataset_info)
    
    logger.info(f"MLflow 데이터셋 목록 조회 완료: {len(datasets)}개")
    
    return create_response(
        success=True,
        message=f"{len(datasets)}개의 데이터셋을 찾았습니다",
        total_count=len(datasets),
        datasets=datasets,
        filter={
            "experiment_name": list_request.experiment_name,
            "user_id": user_id
        }
    )

@router.post("/mlflow/get-dataset-columns", summary="MLflow 데이터셋 컬럼 조회", response_model=Dict[str, Any])
@handle_errors("MLflow 데이터셋 컬럼 조회")
async def get_mlflow_dataset_columns(request: Request, columns_request: GetMLflowDatasetColumnsRequest) -> Dict[str, Any]:
    """MLflow 데이터셋의 컬럼 정보 조회"""
    try:
        import mlflow
        import pyarrow.parquet as pq
        import pyarrow.csv as csv
    except ImportError:
        raise HTTPException(status_code=500, detail="필수 패키지가 설치되지 않았습니다")

    user_id = extract_user_id_from_request(request)
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID가 제공되지 않았습니다")

    # MLflow 설정
    config_composer = get_config_composer(request)
    mlflow_config = get_mlflow_config(config_composer, columns_request.mlflow_tracking_uri)
    setup_mlflow_env(mlflow_config)
    
    client = mlflow.tracking.MlflowClient()
    
    logger.info(f"MLflow 데이터셋 컬럼 조회: run_id={columns_request.run_id}")
    
    # 아티팩트 목록 조회
    artifacts = client.list_artifacts(columns_request.run_id, columns_request.artifact_path)
    
    # 데이터셋 파일 찾기
    dataset_file = None
    for artifact in artifacts:
        if artifact.path.endswith('.csv') or artifact.path.endswith('.parquet'):
            dataset_file = artifact
            break
    
    if not dataset_file:
        raise HTTPException(
            status_code=404,
            detail=f"Run {columns_request.run_id}에서 데이터셋 파일을 찾을 수 없습니다"
        )
    
    # 임시 디렉토리에 파일 다운로드
    with tempfile.TemporaryDirectory() as temp_dir:
        local_path = client.download_artifacts(
            columns_request.run_id,
            dataset_file.path,
            dst_path=temp_dir
        )
        
        # 파일 형식에 따라 컬럼 정보 추출
        file_format = 'parquet' if dataset_file.path.endswith('.parquet') else 'csv'
        
        if file_format == 'parquet':
            table = pq.read_table(local_path)
        else:
            table = csv.read_csv(local_path)
        
        schema = table.schema
        columns_info = [
            {
                "name": field.name,
                "type": str(field.type),
                "nullable": field.nullable
            }
            for field in schema
        ]
        
        num_rows = len(table)
    
    logger.info(f"컬럼 정보 조회 완료: {len(columns_info)}개 컬럼")
    
    return create_response(
        success=True,
        message="컬럼 정보를 성공적으로 조회했습니다",
        run_id=columns_request.run_id,
        file_path=dataset_file.path,
        file_format=file_format,
        num_rows=num_rows,
        num_columns=len(columns_info),
        columns=columns_info
    )

# ========== 수정된 test_database_connection ==========
@router.post("/db/test-connection", summary="DB 연결 테스트", response_model=Dict[str, Any])
async def test_database_connection(request: Request, conn_request: DatabaseConnectionRequest) -> Dict[str, Any]:
    """데이터베이스 연결 테스트 (개선된 에러 처리)"""
    user_id = extract_user_id_from_request(request)
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID가 제공되지 않았습니다")

    # DB 설정 가져오기
    try:
        db_config = get_db_config_from_request(conn_request)
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"DB 설정 파싱 실패: {e}")
        raise HTTPException(status_code=400, detail=f"데이터베이스 설정을 파싱할 수 없습니다: {str(e)}")
    
    try:
        from sqlalchemy import create_engine, text
    except ImportError:
        raise HTTPException(status_code=500, detail="sqlalchemy 패키지가 설치되지 않았습니다")
    
    db_type = db_config.get('db_type', 'postgresql').lower()
    connection_string = None
    engine = None
    
    try:
        # 연결 문자열 생성
        connection_string = create_db_connection_string(db_config)
        logger.info(f"DB 연결 시도: {db_type}://{db_config.get('username', '')}@{db_config.get('host', '')}:{db_config.get('port', '')}/{db_config['database']}")
        
        # 연결 테스트
        engine = create_engine(
            connection_string,
            pool_pre_ping=True,  # 연결 전 핑 테스트
            connect_args={
                'connect_timeout': 10  # 10초 타임아웃
            }
        )
        
        schemas = []
        tables = []
        
        with engine.connect() as conn:
            # 간단한 쿼리로 연결 테스트
            conn.execute(text("SELECT 1"))
            
            # 스키마/테이블 정보 조회
            if db_type == 'postgresql':
                # 스키마 목록
                schemas_query = text("""
                    SELECT schema_name, 
                           (SELECT COUNT(*) FROM information_schema.tables 
                            WHERE table_schema = s.schema_name AND table_type = 'BASE TABLE') as table_count
                    FROM information_schema.schemata s
                    WHERE schema_name NOT IN ('pg_catalog', 'information_schema')
                    ORDER BY schema_name
                """)
                schemas_result = conn.execute(schemas_query)
                schemas = [{"schema_name": row[0], "table_count": row[1]} for row in schemas_result]
                
                # 테이블 목록 (최대 50개)
                tables_query = text("""
                    SELECT table_schema, table_name 
                    FROM information_schema.tables 
                    WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
                    AND table_type = 'BASE TABLE'
                    ORDER BY table_schema, table_name
                    LIMIT 50
                """)
                tables_result = conn.execute(tables_query)
                tables = [{"schema": row[0], "table": row[1]} for row in tables_result]
                
            elif db_type == 'mysql':
                tables_query = text("SHOW TABLES")
                tables_result = conn.execute(tables_query)
                tables = [{"table": row[0]} for row in tables_result]
                
            elif db_type == 'sqlite':
                tables_query = text("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table'
                    ORDER BY name
                """)
                tables_result = conn.execute(tables_query)
                tables = [{"table": row[0]} for row in tables_result]
        
        if engine:
            engine.dispose()
        
        logger.info(f"✅ DB 연결 성공: {db_type}/{db_config['database']}, {len(tables)}개 테이블")
        
        return create_response(
            success=True,
            message="데이터베이스 연결 성공",
            db_type=db_type,
            database=db_config['database'],
            host=db_config.get('host'),
            port=db_config.get('port'),
            connection_status="connected",
            schemas_count=len(schemas) if db_type == 'postgresql' else None,
            schemas=schemas if db_type == 'postgresql' else None,
            tables_count=len(tables),
            tables=tables,
            parsed_config={
                k: v for k, v in db_config.items() 
                if k != 'password'  # 비밀번호 제외
            }
        )
        
    except (OperationalError, DatabaseError) as e:
        # SQLAlchemy 데이터베이스 오류
        if engine:
            engine.dispose()
        
        error_info = parse_db_error(e, db_config)
        logger.error(f"❌ DB 연결 실패 ({error_info['error_type']}): {error_info['user_message']}")
        
        raise HTTPException(
            status_code=400,  # 400으로 변경 (사용자 입력 오류)
            detail={
                "error_type": error_info['error_type'],
                "message": error_info['user_message'],
                "details": error_info['details'],
                "suggestions": error_info['suggestions'],
                "connection_info": {
                    "db_type": db_type,
                    "host": db_config.get('host'),
                    "port": db_config.get('port'),
                    "database": db_config.get('database'),
                    "username": db_config.get('username')
                }
            }
        )
    
    except ImportError as e:
        if engine:
            engine.dispose()
        
        logger.error(f"필수 패키지 미설치: {e}")
        
        # DB 타입별 필요 패키지 안내
        required_packages = {
            'postgresql': 'psycopg2-binary',
            'mysql': 'pymysql',
            'sqlite': '(내장 패키지)'
        }
        
        raise HTTPException(
            status_code=500,
            detail={
                "error_type": "MISSING_PACKAGE",
                "message": f"{db_type} 데이터베이스 연결에 필요한 패키지가 설치되지 않았습니다",
                "required_package": required_packages.get(db_type, 'unknown'),
                "suggestions": [
                    f"다음 명령어로 패키지를 설치하세요:",
                    f"pip install {required_packages.get(db_type, 'required-package')}"
                ]
            }
        )
    
    except Exception as e:
        if engine:
            engine.dispose()
        
        logger.error(f"예상치 못한 오류: {e}", exc_info=True)
        
        raise HTTPException(
            status_code=500,
            detail={
                "error_type": "UNEXPECTED_ERROR",
                "message": "데이터베이스 연결 테스트 중 오류가 발생했습니다",
                "details": str(e)[:200],
                "suggestions": [
                    "1. 모든 연결 설정을 확인하세요",
                    "2. 로그를 확인하여 자세한 정보를 확인하세요",
                    "3. 관리자에게 문의하세요"
                ]
            }
        )

@router.post("/db/list-tables", summary="테이블 목록 조회", response_model=Dict[str, Any])
@handle_errors("테이블 목록 조회")
async def list_database_tables(request: Request, list_request: TableListRequest) -> Dict[str, Any]:
    """데이터베이스 테이블 목록 조회"""
    user_id = extract_user_id_from_request(request)
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID가 제공되지 않았습니다")

    db_config = get_db_config_from_request(list_request)
    
    try:
        from sqlalchemy import create_engine, text, inspect
    except ImportError:
        raise HTTPException(status_code=500, detail="sqlalchemy 패키지가 필요합니다")
    
    connection_string = create_db_connection_string(db_config)
    engine = create_engine(connection_string)
    inspector = inspect(engine)
    
    db_type = db_config.get('db_type', 'postgresql').lower()
    tables_info = []
    
    with engine.connect() as conn:
        if db_type == 'postgresql':
            query = text("""
                SELECT 
                    table_schema,
                    table_name,
                    (SELECT COUNT(*) FROM information_schema.columns 
                     WHERE table_schema = t.table_schema AND table_name = t.table_name) as column_count
                FROM information_schema.tables t
                WHERE (:schema IS NULL OR table_schema = :schema)
                AND table_schema NOT IN ('pg_catalog', 'information_schema')
                AND table_type = 'BASE TABLE'
                ORDER BY table_schema, table_name
            """)
            result = conn.execute(query, {"schema": list_request.schema})
            
            for row in result:
                try:
                    count_query = text(f'SELECT COUNT(*) FROM "{row[0]}"."{row[1]}"')
                    row_count = conn.execute(count_query).scalar()
                except Exception:
                    row_count = None
                
                tables_info.append({
                    "schema": row[0],
                    "table_name": row[1],
                    "column_count": row[2],
                    "row_count": row_count,
                    "full_name": f"{row[0]}.{row[1]}"
                })
        
        elif db_type == 'mysql':
            query = text("""
                SELECT TABLE_SCHEMA, TABLE_NAME,
                       (SELECT COUNT(*) FROM information_schema.COLUMNS 
                        WHERE TABLE_SCHEMA = t.TABLE_SCHEMA AND TABLE_NAME = t.TABLE_NAME) as COLUMN_COUNT,
                       TABLE_ROWS
                FROM information_schema.TABLES t
                WHERE TABLE_SCHEMA = :database AND TABLE_TYPE = 'BASE TABLE'
                ORDER BY TABLE_NAME
            """)
            result = conn.execute(query, {"database": db_config['database']})
            
            for row in result:
                tables_info.append({
                    "schema": row[0],
                    "table_name": row[1],
                    "column_count": row[2],
                    "row_count": row[3],
                    "full_name": f"{row[0]}.{row[1]}"
                })
        
        elif db_type == 'sqlite':
            table_names = inspector.get_table_names()
            
            for table_name in table_names:
                columns = inspector.get_columns(table_name)
                column_count = len(columns)
                
                try:
                    count_query = text(f'SELECT COUNT(*) FROM "{table_name}"')
                    row_count = conn.execute(count_query).scalar()
                except Exception:
                    row_count = None
                
                tables_info.append({
                    "table_name": table_name,
                    "column_count": column_count,
                    "row_count": row_count,
                    "full_name": table_name
                })
    
    engine.dispose()
    
    logger.info(f"테이블 목록 조회 완료: {len(tables_info)}개 테이블")
    
    return create_response(
        success=True,
        message=f"{len(tables_info)}개의 테이블을 찾았습니다",
        db_type=db_type,
        database=db_config['database'],
        schema=list_request.schema,
        table_count=len(tables_info),
        tables=tables_info
    )

@router.post("/db/preview-table", summary="테이블 미리보기", response_model=Dict[str, Any])
@handle_errors("테이블 미리보기")
async def preview_database_table(request: Request, preview_request: TablePreviewRequest) -> Dict[str, Any]:
    """데이터베이스 테이블 미리보기"""
    user_id = extract_user_id_from_request(request)
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID가 제공되지 않았습니다")

    db_config = get_db_config_from_request(preview_request)
    
    try:
        from sqlalchemy import create_engine, text, inspect
        import pandas as pd
    except ImportError:
        raise HTTPException(status_code=500, detail="필수 패키지가 설치되지 않았습니다")
    
    connection_string = create_db_connection_string(db_config)
    engine = create_engine(connection_string)
    inspector = inspect(engine)
    
    db_type = db_config.get('db_type', 'postgresql').lower()
    
    # 테이블 식별자 생성
    if db_type == 'postgresql' and preview_request.schema:
        table_identifier = f'"{preview_request.schema}"."{preview_request.table_name}"'
        schema_param = preview_request.schema
    else:
        table_identifier = f'"{preview_request.table_name}"'
        schema_param = None
    
    # 테이블 존재 확인
    available_tables = inspector.get_table_names(schema=schema_param)
    if preview_request.table_name not in available_tables:
        engine.dispose()
        raise HTTPException(status_code=404, detail=f"테이블 '{preview_request.table_name}'을 찾을 수 없습니다")
    
    # 스키마 정보 조회
    columns = inspector.get_columns(preview_request.table_name, schema=schema_param)
    column_info = [
        {
            "name": col['name'],
            "type": str(col['type']),
            "nullable": col.get('nullable', True),
            "default": str(col.get('default')) if col.get('default') is not None else None
        }
        for col in columns
    ]
    
    # 샘플 데이터 조회
    with engine.connect() as conn:
        query = f"SELECT * FROM {table_identifier} LIMIT {preview_request.limit}"
        df = pd.read_sql(query, conn)
        sample_data = df.to_dict('records')
        
        # 전체 행 수 조회
        count_query = f"SELECT COUNT(*) as total FROM {table_identifier}"
        total_rows = pd.read_sql(count_query, conn)['total'][0]
    
    engine.dispose()
    
    logger.info(f"테이블 미리보기: {preview_request.table_name} ({total_rows} rows)")
    
    return create_response(
        success=True,
        message="테이블 미리보기 성공",
        table_name=preview_request.table_name,
        schema=preview_request.schema,
        total_rows=int(total_rows),
        total_columns=len(column_info),
        columns=column_info,
        sample_data=sample_data,
        sample_count=len(sample_data)
    )

@router.post("/db/validate-query", summary="SQL 쿼리 검증", response_model=Dict[str, Any])
@handle_errors("쿼리 검증")
async def validate_sql_query(request: Request, validation_request: QueryValidationRequest) -> Dict[str, Any]:
    """SQL 쿼리 유효성 검증"""
    user_id = extract_user_id_from_request(request)
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID가 제공되지 않았습니다")

    db_config = get_db_config_from_request(validation_request)
    
    try:
        from sqlalchemy import create_engine, text
        import pandas as pd
    except ImportError:
        raise HTTPException(status_code=500, detail="필수 패키지가 설치되지 않았습니다")
    
    connection_string = create_db_connection_string(db_config)
    engine = create_engine(connection_string)
    
    # 쿼리 정리
    test_query = validation_request.query.strip().rstrip(';')
    
    # LIMIT 추가 (없는 경우)
    if 'LIMIT' not in test_query.upper():
        test_query = f"{test_query} LIMIT 1"
    
    try:
        with engine.connect() as conn:
            # 쿼리 실행 (1개 행만)
            df = pd.read_sql(test_query, conn)
            
            # 컬럼 정보 추출
            column_info = [
                {
                    "name": col,
                    "type": str(df[col].dtype),
                    "sample_value": str(df[col].iloc[0]) if len(df) > 0 else None
                }
                for col in df.columns
            ]
            
            # 예상 행 수 추정
            estimated_rows = None
            try:
                original_query = validation_request.query.strip().rstrip(';')
                count_query = f"SELECT COUNT(*) as total FROM ({original_query}) as subquery"
                count_df = pd.read_sql(count_query, conn)
                estimated_rows = int(count_df['total'][0])
            except Exception as count_error:
                logger.warning(f"행 수 추정 실패: {count_error}")
        
        engine.dispose()
        
        logger.info(f"쿼리 검증 성공: {len(column_info)}개 컬럼, 예상 {estimated_rows}행")
        
        return {
            "success": True,
            "valid": True,
            "message": "쿼리가 유효합니다",
            "column_count": len(column_info),
            "columns": column_info,
            "estimated_rows": estimated_rows,
            "query": validation_request.query
        }
        
    except Exception as query_error:
        engine.dispose()
        error_message = str(query_error)
        logger.warning(f"쿼리 검증 실패: {error_message}")
        
        return {
            "success": False,
            "valid": False,
            "message": "쿼리 실행 실패",
            "error": error_message,
            "query": validation_request.query
        }

@router.post("/db/load-dataset", summary="DB에서 데이터셋 로드", response_model=Dict[str, Any])
@handle_errors("DB 데이터셋 로드")
async def load_dataset_from_database(request: Request, load_request: LoadFromDatabaseRequest) -> Dict[str, Any]:
    """데이터베이스에서 데이터셋 로드"""
    user_id = extract_user_id_from_request(request)
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID가 제공되지 않았습니다")

    registry = get_data_manager_registry(request)
    manager = registry.get_manager(load_request.manager_id, user_id)
    if not manager:
        raise HTTPException(status_code=404, detail="매니저를 찾을 수 없습니다")

    # DB 설정 가져오기
    db_config = load_request.db_config
    if load_request.connection_url:
        try:
            db_config = parse_connection_url(load_request.connection_url)
            logger.info(f"URL에서 DB 설정 파싱: {db_config['db_type']}/{db_config['database']}")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    if not db_config:
        raise HTTPException(status_code=400, detail="db_config 또는 connection_url 중 하나가 필요합니다")

    validate_db_config(db_config)

    # query 또는 table_name 필수
    if not load_request.query and not load_request.table_name:
        raise HTTPException(status_code=400, detail="query 또는 table_name 중 하나는 필수입니다")

    # 데이터베이스에서 로드
    result = manager.db_load_dataset(
        db_config=db_config,
        query=load_request.query,
        table_name=load_request.table_name,
        chunk_size=load_request.chunk_size
    )

    logger.info(f"DB 로드 완료: {load_request.manager_id} from {db_config['db_type']}/{db_config['database']}")

    return create_response(
        success=True,
        message=f"데이터베이스에서 데이터셋이 로드되었습니다 (v{result['load_count']})",
        manager_id=load_request.manager_id,
        load_info=result,
        version_info={
            "dataset_id": result.get("dataset_id"),
            "load_count": result.get("load_count"),
            "is_new_version": result.get("is_new_version", False)
        }
    )

# ========== Health Check ==========
@router.get("/health", summary="헬스 체크")
async def health_check():
    """API 상태 확인"""
    return {"status": "healthy", "service": "data-manager-processing"}
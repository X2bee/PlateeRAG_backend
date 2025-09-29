from fastapi import APIRouter, HTTPException, Request, File, UploadFile, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import logging
import os
from typing import Dict, Any, Optional, List
from datetime import datetime
from controller.helper.singletonHelper import get_db_manager, get_data_manager_registry, get_config_composer
from controller.helper.controllerHelper import extract_user_id_from_request
import collections
import collections.abc
if not hasattr(collections, 'Callable'):
    collections.Callable = collections.abc.Callable

router = APIRouter(prefix="/processing", tags=["data-manager"])
logger = logging.getLogger("data-manager-controller")

# ========== Request Models ==========
class DownloadDatasetRequest(BaseModel):
    """데이터셋 다운로드 요청"""
    manager_id: str = Field(..., description="매니저 ID")
    repo_id: str = Field(..., description="Huggingface 리포지토리 ID", example="squad")
    filename: Optional[str] = Field(None, description="특정 파일명 (선택사항)")
    split: Optional[str] = Field(None, description="데이터 분할", example="train")

class ExportDatasetRequest(BaseModel):
    """데이터셋 내보내기 요청"""
    manager_id: str = Field(..., description="매니저 ID")

class DropColumnsRequest(BaseModel):
    """컬럼 삭제 요청"""
    manager_id: str = Field(..., description="매니저 ID")
    columns: List[str] = Field(..., description="삭제할 컬럼명들", min_items=1)

class ReplaceValuesRequest(BaseModel):
    """값 교체 요청"""
    manager_id: str = Field(..., description="매니저 ID")
    column_name: str = Field(..., description="대상 컬럼명")
    old_value: str = Field(..., description="교체할 기존 값")
    new_value: str = Field(..., description="새로운 값")

class ApplyOperationRequest(BaseModel):
    """연산 적용 요청"""
    manager_id: str = Field(..., description="매니저 ID")
    column_name: str = Field(..., description="대상 컬럼명")
    operation: str = Field(..., description="연산식 (예: +4, *3+4)", pattern=r'^[+\-*/\d.]+$')

class RemoveNullRowsRequest(BaseModel):
    """NULL row 제거 요청"""
    manager_id: str = Field(..., description="매니저 ID")
    column_name: Optional[str] = Field(None, description="특정 컬럼명 (미지정시 전체 컬럼에서 NULL 체크)")

class UploadToHfRequest(BaseModel):
    """HuggingFace Hub 업로드 요청"""
    manager_id: str = Field(..., description="매니저 ID")
    repo_id: str = Field(..., description="HuggingFace 리포지토리 ID (user/repo-name 또는 repo-name)")
    filename: Optional[str] = Field(None, description="업로드할 파일명 (미지정시 자동 생성)")
    private: bool = Field(False, description="프라이빗 리포지토리 여부")
    hf_user_id: Optional[str] = Field(None, description="HuggingFace 사용자 ID (미지정시 설정값 사용)")
    hub_token: Optional[str] = Field(None, description="HuggingFace Hub 토큰 (미지정시 설정값 사용)")

class CopyColumnRequest(BaseModel):
    """컬럼 복사 요청"""
    manager_id: str = Field(..., description="매니저 ID")
    source_column: str = Field(..., description="복사할 원본 컬럼명")
    new_column: str = Field(..., description="새로운 컬럼명")

class RenameColumnRequest(BaseModel):
    """컬럼 이름 변경 요청"""
    manager_id: str = Field(..., description="매니저 ID")
    old_name: str = Field(..., description="기존 컬럼명")
    new_name: str = Field(..., description="새로운 컬럼명")

class FormatColumnsRequest(BaseModel):
    """컬럼 문자열 포맷팅 요청"""
    manager_id: str = Field(..., description="매니저 ID")
    column_names: List[str] = Field(..., description="사용할 컬럼명들", min_items=1)
    template: str = Field(..., description="문자열 템플릿 (예: {col1}_aiaiaiai_{col2})")
    new_column: str = Field(..., description="새로운 컬럼명")

class CalculateColumnsRequest(BaseModel):
    """컬럼 간 사칙연산 요청"""
    manager_id: str = Field(..., description="매니저 ID")
    col1: str = Field(..., description="첫 번째 컬럼명")
    col2: str = Field(..., description="두 번째 컬럼명")
    operation: str = Field(..., description="연산자", pattern=r'^[+\-*/]$')
    new_column: str = Field(..., description="새로운 컬럼명")

class ExecuteCallbackRequest(BaseModel):
    """사용자 콜백 코드 실행 요청"""
    manager_id: str = Field(..., description="매니저 ID")
    callback_code: str = Field(..., description="실행할 PyArrow 코드", min_length=1)

class UploadToMlflowRequest(BaseModel):
    """MLflow에 데이터셋 업로드 요청"""
    manager_id: str = Field(..., description="매니저 ID")
    experiment_name: str = Field(..., description="MLflow 실험 이름")
    artifact_path: str = Field("dataset", description="아티팩트 저장 경로")
    dataset_name: str = Field(..., description="데이터셋 이름")
    description: Optional[str] = Field(None, description="데이터셋 설명")
    tags: Optional[Dict[str, str]] = Field(None, description="추가 태그")
    format: str = Field("parquet", description="저장 형식", pattern="^(parquet|csv)$")
    mlflow_tracking_uri: Optional[str] = Field(None, description="MLflow 추적 서버 URI (선택사항)")

class ListMlflowDatasetsRequest(BaseModel):
    """MLflow 데이터셋 목록 조회 요청"""
    experiment_name: Optional[str] = Field(None, description="특정 실험명으로 필터링 (선택사항)")
    max_results: int = Field(100, description="최대 결과 개수", ge=1, le=1000)
    mlflow_tracking_uri: Optional[str] = Field(None, description="MLflow 추적 서버 URI (선택사항)")

class GetMLflowDatasetColumnsRequest(BaseModel):
    """MLflow 데이터셋 컬럼 조회 요청"""
    run_id: str = Field(..., description="MLflow Run ID")
    artifact_path: Optional[str] = Field("dataset", description="아티팩트 경로")
    mlflow_tracking_uri: Optional[str] = Field(None, description="MLflow 추적 서버 URI")

# ========== Helper Functions ==========
def get_manager_with_auth(registry, manager_id: str, user_id: str):
    """인증된 매니저 가져오기"""
    manager = registry.get_manager(manager_id, user_id)
    if not manager:
        raise HTTPException(
            status_code=404,
            detail=f"매니저 '{manager_id}'를 찾을 수 없거나 접근 권한이 없습니다"
        )
    return manager

# ========== API Endpoints ==========

@router.post("/hf/download-dataset",
    summary="데이터셋 다운로드 및 적재",
    description="Huggingface 리포지토리에서 데이터셋을 다운로드하고 pyarrow로 적재합니다.",
    response_model=Dict[str, Any])
async def hf_download_dataset(request: Request, download_request: DownloadDatasetRequest) -> Dict[str, Any]:
    """데이터셋 다운로드 및 적재"""
    try:
        user_id = extract_user_id_from_request(request)
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID가 제공되지 않았습니다")

        registry = get_data_manager_registry(request)
        manager = get_manager_with_auth(registry, download_request.manager_id, user_id)

        # 데이터셋 다운로드 및 적재
        result = manager.hf_download_and_load_dataset(
            repo_id=download_request.repo_id,
            filename=download_request.filename,
            split=download_request.split
        )

        logger.info(f"Dataset downloaded and loaded for manager {download_request.manager_id} from {download_request.repo_id}")

        return {
            "success": True,
            "manager_id": download_request.manager_id,
            "message": "데이터셋이 성공적으로 다운로드되고 적재되었습니다",
            "download_info": result
        }

    except HTTPException:
        raise
    except RuntimeError as e:
        logger.error(f"데이터셋 다운로드 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"예상치 못한 오류: {e}")
        raise HTTPException(status_code=500, detail="데이터셋 다운로드 중 오류가 발생했습니다")


@router.post("/local/upload-dataset",
    summary="로컬 파일 업로드 및 자동 적재",
    description="로컬 파일들을 업로드하고 즉시 적재. 다중 파일 지원.",
    response_model=Dict[str, Any])
async def local_upload_dataset(
    request: Request,
    files: List[UploadFile] = File(..., description="업로드할 데이터셋 파일들 (parquet 또는 csv)"),
    manager_id: str = Form(..., description="매니저 ID")
) -> Dict[str, Any]:
    """로컬 파일 업로드 및 자동 적재"""
    try:
        user_id = extract_user_id_from_request(request)
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID가 제공되지 않았습니다")

        registry = get_data_manager_registry(request)
        manager = get_manager_with_auth(registry, manager_id, user_id)

        # 파일 형식 검증 - 모든 파일이 같은 형식이어야 함
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
                detail=f"지원되지 않는 파일 형식입니다. parquet 또는 csv 파일만 지원됩니다. (업로드된 형식: {file_type})"
            )

        # 파일들과 이름들 준비
        uploaded_files = [file.file for file in files]
        filenames = [file.filename for file in files]

        # 업로드 및 자동 적재
        result = manager.local_upload_and_load_dataset(uploaded_files, filenames)

        logger.info("로컬 데이터셋 업로드 완료: 매니저 %s, 파일 %d개", manager_id, len(files))

        return {
            "success": True,
            "manager_id": manager_id,
            "message": f"{len(files)}개 파일이 성공적으로 업로드되고 적재되었습니다",
            "dataset_info": result
        }

    except HTTPException:
        raise
    except RuntimeError as e:
        logger.error("로컬 데이터셋 업로드 실패: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error("예상치 못한 오류: %s", e)
        raise HTTPException(status_code=500, detail="로컬 데이터셋 업로드 중 오류가 발생했습니다")


@router.post("/export/csv",
    summary="데이터셋을 CSV로 다운로드",
    description="현재 적재된 데이터셋을 CSV 파일로 내보내기")
async def export_dataset_as_csv(request: Request, export_request: ExportDatasetRequest):
    """데이터셋을 CSV 파일로 다운로드"""
    try:
        user_id = extract_user_id_from_request(request)
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID가 제공되지 않았습니다")

        registry = get_data_manager_registry(request)
        manager = get_manager_with_auth(registry, export_request.manager_id, user_id)

        # CSV 파일 생성
        file_path = manager.download_dataset_as_csv()

        # 파일 이름 설정
        filename = f"dataset_{export_request.manager_id}.csv"

        logger.info("CSV 다운로드 요청: 매니저 %s", export_request.manager_id)

        return FileResponse(
            path=file_path,
            filename=filename,
            media_type='text/csv',
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )

    except HTTPException:
        raise
    except RuntimeError as e:
        logger.error("CSV 내보내기 실패: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error("예상치 못한 오류: %s", e)
        raise HTTPException(status_code=500, detail="CSV 내보내기 중 오류가 발생했습니다")


@router.post("/export/parquet",
    summary="데이터셋을 Parquet으로 다운로드",
    description="현재 적재된 데이터셋을 Parquet 파일로 내보내기")
async def export_dataset_as_parquet(request: Request, export_request: ExportDatasetRequest):
    """데이터셋을 Parquet 파일로 다운로드"""
    try:
        user_id = extract_user_id_from_request(request)
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID가 제공되지 않았습니다")

        registry = get_data_manager_registry(request)
        manager = get_manager_with_auth(registry, export_request.manager_id, user_id)

        # Parquet 파일 생성
        file_path = manager.download_dataset_as_parquet()

        # 파일 이름 설정
        filename = f"dataset_{export_request.manager_id}.parquet"

        logger.info("Parquet 다운로드 요청: 매니저 %s", export_request.manager_id)

        return FileResponse(
            path=file_path,
            filename=filename,
            media_type='application/octet-stream',
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )

    except HTTPException:
        raise
    except RuntimeError as e:
        logger.error("Parquet 내보내기 실패: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error("예상치 못한 오류: %s", e)
        raise HTTPException(status_code=500, detail="Parquet 내보내기 중 오류가 발생했습니다")


@router.post("/statistics",
    summary="데이터셋 기술통계정보 조회",
    description="현재 적재된 데이터셋의 기술통계정보를 반환합니다.",
    response_model=Dict[str, Any])
async def get_dataset_statistics(request: Request, export_request: ExportDatasetRequest) -> Dict[str, Any]:
    """데이터셋 기술통계정보 조회"""
    try:
        user_id = extract_user_id_from_request(request)
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID가 제공되지 않았습니다")

        registry = get_data_manager_registry(request)
        manager = get_manager_with_auth(registry, export_request.manager_id, user_id)

        # 기술통계정보 생성
        statistics = manager.get_dataset_statistics()

        logger.info("통계정보 조회: 매니저 %s", export_request.manager_id)

        return {
            "success": True,
            "manager_id": export_request.manager_id,
            "message": "데이터셋 기술통계정보가 성공적으로 생성되었습니다",
            "statistics": statistics
        }

    except HTTPException:
        raise
    except RuntimeError as e:
        logger.error("통계정보 생성 실패: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error("예상치 못한 오류: %s", e)
        raise HTTPException(status_code=500, detail="통계정보 생성 중 오류가 발생했습니다")


@router.post("/drop-columns",
    summary="데이터셋 컬럼 삭제",
    description="현재 적재된 데이터셋에서 지정된 컬럼들을 삭제합니다.",
    response_model=Dict[str, Any])
async def drop_dataset_columns(request: Request, drop_request: DropColumnsRequest) -> Dict[str, Any]:
    """데이터셋 컬럼 삭제"""
    try:
        user_id = extract_user_id_from_request(request)
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID가 제공되지 않았습니다")

        registry = get_data_manager_registry(request)
        manager = get_manager_with_auth(registry, drop_request.manager_id, user_id)

        # 컬럼 삭제 실행
        result_info = manager.drop_dataset_columns(drop_request.columns)

        logger.info("컬럼 삭제 완료: 매니저 %s, 삭제된 컬럼: %s",
                   drop_request.manager_id, drop_request.columns)

        return {
            "success": True,
            "manager_id": drop_request.manager_id,
            "message": f"{len(drop_request.columns)}개 컬럼이 성공적으로 삭제되었습니다",
            "drop_info": result_info
        }

    except HTTPException:
        raise
    except RuntimeError as e:
        logger.error("컬럼 삭제 실패: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("예상치 못한 오류: %s", e)
        raise HTTPException(status_code=500, detail="컬럼 삭제 중 오류가 발생했습니다")


@router.post("/replace-values",
    summary="컬럼 값 교체",
    description="특정 컬럼에서 문자열 값을 다른 값으로 교체합니다.",
    response_model=Dict[str, Any])
async def replace_column_values(request: Request, replace_request: ReplaceValuesRequest) -> Dict[str, Any]:
    """컬럼 값 교체"""
    try:
        user_id = extract_user_id_from_request(request)
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID가 제공되지 않았습니다")

        registry = get_data_manager_registry(request)
        manager = get_manager_with_auth(registry, replace_request.manager_id, user_id)

        # 값 교체 실행
        result_info = manager.replace_dataset_column_values(
            replace_request.column_name,
            replace_request.old_value,
            replace_request.new_value
        )

        logger.info("값 교체 완료: 매니저 %s, 컬럼 %s",
                   replace_request.manager_id, replace_request.column_name)

        return {
            "success": True,
            "manager_id": replace_request.manager_id,
            "message": f"컬럼 '{replace_request.column_name}'에서 값이 성공적으로 교체되었습니다",
            "replace_info": result_info
        }

    except HTTPException:
        raise
    except RuntimeError as e:
        logger.error("값 교체 실패: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("예상치 못한 오류: %s", e)
        raise HTTPException(status_code=500, detail="값 교체 중 오류가 발생했습니다")


@router.post("/apply-operation",
    summary="컬럼 연산 적용",
    description="특정 컬럼에 수치 연산을 적용합니다.",
    response_model=Dict[str, Any])
async def apply_column_operation(request: Request, operation_request: ApplyOperationRequest) -> Dict[str, Any]:
    """컬럼 연산 적용"""
    try:
        user_id = extract_user_id_from_request(request)
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID가 제공되지 않았습니다")

        registry = get_data_manager_registry(request)
        manager = get_manager_with_auth(registry, operation_request.manager_id, user_id)

        # 연산 적용 실행
        result_info = manager.apply_dataset_column_operation(
            operation_request.column_name,
            operation_request.operation
        )

        logger.info("연산 적용 완료: 매니저 %s, 컬럼 %s, 연산 %s",
                   operation_request.manager_id, operation_request.column_name, operation_request.operation)

        return {
            "success": True,
            "manager_id": operation_request.manager_id,
            "message": f"컬럼 '{operation_request.column_name}'에 연산 '{operation_request.operation}'이 성공적으로 적용되었습니다",
            "operation_info": result_info
        }

    except HTTPException:
        raise
    except RuntimeError as e:
        logger.error("연산 적용 실패: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("예상치 못한 오류: %s", e)
        raise HTTPException(status_code=500, detail="연산 적용 중 오류가 발생했습니다")


@router.post("/remove-null-rows",
    summary="NULL row 제거",
    description="NULL 값이 있는 행을 제거합니다. 특정 컬럼 지정 가능.",
    response_model=Dict[str, Any])
async def remove_null_rows(request: Request, remove_request: RemoveNullRowsRequest) -> Dict[str, Any]:
    """NULL 값이 있는 행 제거"""
    try:
        user_id = extract_user_id_from_request(request)
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID가 제공되지 않았습니다")

        registry = get_data_manager_registry(request)
        manager = get_manager_with_auth(registry, remove_request.manager_id, user_id)

        # NULL row 제거 실행
        result_info = manager.remove_null_rows_from_dataset(remove_request.column_name)

        # 로그 메시지 생성
        if remove_request.column_name:
            log_msg = f"매니저 {remove_request.manager_id}, 컬럼 '{remove_request.column_name}'"
            response_msg = f"컬럼 '{remove_request.column_name}'에서 NULL 값이 있는 행이 성공적으로 제거되었습니다"
        else:
            log_msg = f"매니저 {remove_request.manager_id}, 전체 컬럼"
            response_msg = "NULL 값이 있는 행이 성공적으로 제거되었습니다"

        logger.info("NULL row 제거 완료: %s, %d개 행 제거",
                   log_msg, result_info["removed_rows"])

        return {
            "success": True,
            "manager_id": remove_request.manager_id,
            "message": response_msg,
            "removal_info": result_info
        }

    except HTTPException:
        raise
    except RuntimeError as e:
        logger.error("NULL row 제거 실패: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("예상치 못한 오류: %s", e)
        raise HTTPException(status_code=500, detail="NULL row 제거 중 오류가 발생했습니다")


@router.post("/upload-to-hf",
    summary="HuggingFace Hub에 데이터셋 업로드",
    description="현재 데이터셋을 parquet 파일로 변환하여 HuggingFace Hub에 업로드합니다.",
    response_model=Dict[str, Any])
async def upload_dataset_to_hf(request: Request, upload_request: UploadToHfRequest) -> Dict[str, Any]:
    """데이터셋을 HuggingFace Hub에 업로드"""
    try:
        user_id = extract_user_id_from_request(request)
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID가 제공되지 않았습니다")

        registry = get_data_manager_registry(request)
        manager = get_manager_with_auth(registry, upload_request.manager_id, user_id)

        # config_composer에서 HuggingFace 설정 가져오기
        config_composer = get_config_composer(request)

        # HuggingFace 설정값 결정
        hf_user_id = upload_request.hf_user_id
        hub_token = upload_request.hub_token

        if not hf_user_id:
            try:
                hf_user_id = config_composer.get_config_by_name("HUGGING_FACE_USER_ID").value
            except Exception as e:
                logger.warning("HUGGING_FACE_USER_ID 설정 가져오기 실패: %s", e)
                raise HTTPException(status_code=400, detail="HuggingFace User ID가 제공되지 않았고, 설정값도 찾을 수 없습니다")

        if not hub_token:
            try:
                hub_token = config_composer.get_config_by_name("HUGGING_FACE_HUB_TOKEN").value
            except Exception as e:
                logger.warning("HUGGING_FACE_HUB_TOKEN 설정 가져오기 실패: %s", e)
                raise HTTPException(status_code=400, detail="HuggingFace Hub Token이 제공되지 않았고, 설정값도 찾을 수 없습니다")

        # HuggingFace 업로드 실행
        result_info = manager.upload_dataset_to_hf_repo(
            repo_id=upload_request.repo_id,
            hf_user_id=hf_user_id,
            hub_token=hub_token,
            filename=upload_request.filename,
            private=upload_request.private
        )

        logger.info("HuggingFace 업로드 완료: 매니저 %s → %s",
                   upload_request.manager_id, result_info["repo_id"])

        return {
            "success": True,
            "manager_id": upload_request.manager_id,
            "message": f"데이터셋이 HuggingFace Hub에 성공적으로 업로드되었습니다",
            "upload_info": result_info
        }

    except HTTPException:
        raise
    except RuntimeError as e:
        logger.error("HuggingFace 업로드 실패: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("예상치 못한 오류: %s", e)
        raise HTTPException(status_code=500, detail="HuggingFace 업로드 중 오류가 발생했습니다")


@router.post("/copy-column",
    summary="컬럼 복사",
    description="특정 컬럼을 복사하여 새로운 컬럼으로 추가합니다.",
    response_model=Dict[str, Any])
async def copy_dataset_column(request: Request, copy_request: CopyColumnRequest) -> Dict[str, Any]:
    """컬럼 복사"""
    try:
        user_id = extract_user_id_from_request(request)
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID가 제공되지 않았습니다")

        registry = get_data_manager_registry(request)
        manager = get_manager_with_auth(registry, copy_request.manager_id, user_id)

        # 컬럼 복사 실행
        result_info = manager.copy_dataset_column(
            copy_request.source_column,
            copy_request.new_column
        )

        logger.info("컬럼 복사 완료: 매니저 %s, '%s' → '%s'",
                   copy_request.manager_id, copy_request.source_column, copy_request.new_column)

        return {
            "success": True,
            "manager_id": copy_request.manager_id,
            "message": f"컬럼 '{copy_request.source_column}'이 '{copy_request.new_column}'으로 성공적으로 복사되었습니다",
            "copy_info": result_info
        }

    except HTTPException:
        raise
    except RuntimeError as e:
        logger.error("컬럼 복사 실패: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("예상치 못한 오류: %s", e)
        raise HTTPException(status_code=500, detail="컬럼 복사 중 오류가 발생했습니다")


@router.post("/rename-column",
    summary="컬럼 이름 변경",
    description="특정 컬럼의 이름을 변경합니다.",
    response_model=Dict[str, Any])
async def rename_dataset_column(request: Request, rename_request: RenameColumnRequest) -> Dict[str, Any]:
    """컬럼 이름 변경"""
    try:
        user_id = extract_user_id_from_request(request)
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID가 제공되지 않았습니다")

        registry = get_data_manager_registry(request)
        manager = get_manager_with_auth(registry, rename_request.manager_id, user_id)

        # 컬럼 이름 변경 실행
        result_info = manager.rename_dataset_column(
            rename_request.old_name,
            rename_request.new_name
        )

        logger.info("컬럼 이름 변경 완료: 매니저 %s, '%s' → '%s'",
                   rename_request.manager_id, rename_request.old_name, rename_request.new_name)

        return {
            "success": True,
            "manager_id": rename_request.manager_id,
            "message": f"컬럼 이름이 '{rename_request.old_name}'에서 '{rename_request.new_name}'으로 성공적으로 변경되었습니다",
            "rename_info": result_info
        }

    except HTTPException:
        raise
    except RuntimeError as e:
        logger.error("컬럼 이름 변경 실패: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("예상치 못한 오류: %s", e)
        raise HTTPException(status_code=500, detail="컬럼 이름 변경 중 오류가 발생했습니다")


@router.post("/format-columns",
    summary="컬럼 문자열 포맷팅",
    description="여러 컬럼의 값들을 문자열 템플릿에 삽입하여 새로운 컬럼을 생성합니다.",
    response_model=Dict[str, Any])
async def format_dataset_columns(request: Request, format_request: FormatColumnsRequest) -> Dict[str, Any]:
    """컬럼 문자열 포맷팅"""
    try:
        user_id = extract_user_id_from_request(request)
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID가 제공되지 않았습니다")

        registry = get_data_manager_registry(request)
        manager = get_manager_with_auth(registry, format_request.manager_id, user_id)

        # 컬럼 문자열 포맷팅 실행
        result_info = manager.format_columns_to_string(
            format_request.column_names,
            format_request.template,
            format_request.new_column
        )

        logger.info("컬럼 문자열 포맷팅 완료: 매니저 %s, %s → '%s'",
                   format_request.manager_id, format_request.column_names, format_request.new_column)

        return {
            "success": True,
            "manager_id": format_request.manager_id,
            "message": f"컬럼들 {format_request.column_names}을 템플릿 '{format_request.template}'로 포맷팅하여 새로운 컬럼 '{format_request.new_column}'이 생성되었습니다",
            "format_info": result_info
        }

    except HTTPException:
        raise
    except RuntimeError as e:
        logger.error("컬럼 문자열 포맷팅 실패: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("예상치 못한 오류: %s", e)
        raise HTTPException(status_code=500, detail="컬럼 문자열 포맷팅 중 오류가 발생했습니다")


@router.post("/calculate-columns",
    summary="컬럼 간 사칙연산",
    description="두 컬럼 간 사칙연산을 수행하여 새로운 컬럼을 생성합니다. 문자열과 숫자 타입에 따른 특별 처리 포함.",
    response_model=Dict[str, Any])
async def calculate_dataset_columns(request: Request, calc_request: CalculateColumnsRequest) -> Dict[str, Any]:
    """컬럼 간 사칙연산"""
    try:
        user_id = extract_user_id_from_request(request)
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID가 제공되지 않았습니다")

        registry = get_data_manager_registry(request)
        manager = get_manager_with_auth(registry, calc_request.manager_id, user_id)

        # 컬럼 간 연산 실행
        result_info = manager.calculate_columns_to_new(
            calc_request.col1,
            calc_request.col2,
            calc_request.operation,
            calc_request.new_column
        )

        logger.info("컬럼 간 연산 완료: 매니저 %s, %s %s %s → '%s'",
                   calc_request.manager_id, calc_request.col1, calc_request.operation,
                   calc_request.col2, calc_request.new_column)

        return {
            "success": True,
            "manager_id": calc_request.manager_id,
            "message": f"컬럼 '{calc_request.col1}' {calc_request.operation} '{calc_request.col2}' 연산이 수행되어 새로운 컬럼 '{calc_request.new_column}'이 생성되었습니다",
            "calculation_info": result_info
        }

    except HTTPException:
        raise
    except RuntimeError as e:
        logger.error("컬럼 간 연산 실패: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("예상치 못한 오류: %s", e)
        raise HTTPException(status_code=500, detail="컬럼 간 연산 중 오류가 발생했습니다")


@router.post("/execute-callback",
    summary="사용자 콜백 코드 실행",
    description="사용자가 작성한 PyArrow 코드를 안전하게 실행하여 dataset을 조작합니다. 허용된 PyArrow 함수만 사용 가능합니다.",
    response_model=Dict[str, Any])
async def execute_dataset_callback(request: Request, callback_request: ExecuteCallbackRequest) -> Dict[str, Any]:
    """사용자 콜백 코드 실행"""
    try:
        user_id = extract_user_id_from_request(request)
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID가 제공되지 않았습니다")

        registry = get_data_manager_registry(request)
        manager = get_manager_with_auth(registry, callback_request.manager_id, user_id)

        # 콜백 코드 실행
        result_info = manager.execute_dataset_callback(callback_request.callback_code)

        logger.info("사용자 콜백 실행 완료: 매니저 %s, %d행 → %d행, %d열 → %d열",
                   callback_request.manager_id, result_info["original_rows"],
                   result_info["final_rows"], result_info["original_columns"],
                   result_info["final_columns"])

        return {
            "success": True,
            "manager_id": callback_request.manager_id,
            "message": f"사용자 콜백 코드가 성공적으로 실행되었습니다. {result_info['rows_changed']:+d}행, {result_info['columns_changed']:+d}열 변경",
            "callback_result": result_info
        }

    except HTTPException:
        raise
    except RuntimeError as e:
        logger.error("사용자 콜백 실행 실패: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("예상치 못한 오류: %s", e)
        raise HTTPException(status_code=500, detail="사용자 콜백 실행 중 오류가 발생했습니다")

@router.post("/upload-to-mlflow",
    summary="MLflow에 데이터셋 업로드",
    description="현재 데이터셋을 MLflow 실험의 아티팩트로 업로드합니다.",
    response_model=Dict[str, Any])
async def upload_dataset_to_mlflow(request: Request, upload_request: UploadToMlflowRequest) -> Dict[str, Any]:
    """데이터셋을 MLflow에 업로드"""
    try:
        import mlflow
        import tempfile
        import os
        import json
        
        user_id = extract_user_id_from_request(request)
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID가 제공되지 않았습니다")

        registry = get_data_manager_registry(request)
        manager = get_manager_with_auth(registry, upload_request.manager_id, user_id)

        # S3/MinIO 설정 하드코딩
        os.environ['AWS_ACCESS_KEY_ID'] = 'minioadmin'
        os.environ['AWS_SECRET_ACCESS_KEY'] = 'minioadmin123'
        os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'https://minio.x2bee.com'  # 또는 http://
        
        # MLflow Tracking URI 설정
        if upload_request.mlflow_tracking_uri:
            tracking_uri = upload_request.mlflow_tracking_uri
        else:
            try:
                config_composer = get_config_composer(request)
                tracking_uri = config_composer.get_config_by_name("MLFLOW_TRACKING_URI").value
            except Exception as e:
                logger.warning("MLFLOW_TRACKING_URI 설정 가져오기 실패: %s", e)
                tracking_uri = "https://polar-mlflow-git.x2bee.com"
        
        mlflow.set_tracking_uri(tracking_uri)
        
        logger.info("=== MLflow 설정 ===")
        logger.info(f"Tracking URI: {tracking_uri}")
        logger.info(f"S3 Endpoint: https://minio.x2bee.com")

        # MLflow Client 생성
        client = mlflow.tracking.MlflowClient()

        # 실험 설정
        mlflow.set_experiment(upload_request.experiment_name)
        
        with mlflow.start_run() as run:
            logger.info(f"MLflow Run 시작: run_id={run.info.run_id}, experiment={upload_request.experiment_name}")
            logger.info(f"Artifact URI: {run.info.artifact_uri}")
            
            # S3 사용 확인
            if run.info.artifact_uri.startswith("s3://"):
                logger.info("✅ S3(MinIO) 스토리지 사용 중")
            else:
                logger.warning(f"⚠️  예상치 못한 artifact URI: {run.info.artifact_uri}")
            
            # 데이터셋 파일 생성
            if upload_request.format == "parquet":
                file_path = manager.download_dataset_as_parquet()
            else:
                file_path = manager.download_dataset_as_csv()
            
            file_size = os.path.getsize(file_path)
            logger.info(f"데이터셋 파일 생성: {os.path.basename(file_path)}, 크기: {file_size} bytes")
            
            # 아티팩트 업로드 (S3로 자동 전송)
            logger.info("아티팩트 업로드 시작")
            mlflow.log_artifact(file_path, artifact_path=upload_request.artifact_path)
            logger.info("데이터셋 아티팩트 업로드 완료")
            
            # 메타데이터 로깅
            statistics = manager.get_dataset_statistics()
            
            # 기본 메트릭 로깅
            if isinstance(statistics.get('basic_info'), dict):
                basic_info = statistics['basic_info']
                mlflow.log_metric("dataset_rows", basic_info.get('num_rows', 0))
                mlflow.log_metric("dataset_columns", basic_info.get('num_columns', 0))
                mlflow.log_metric("dataset_size_mb", basic_info.get('memory_usage_mb', 0))
            
            # 파라미터 로깅
            mlflow.log_param("dataset_name", upload_request.dataset_name)
            mlflow.log_param("format", upload_request.format)
            mlflow.log_param("user_id", user_id)
            if upload_request.description:
                mlflow.log_param("description", upload_request.description)
            
            # 태그 설정
            default_tags = {
                "dataset_name": upload_request.dataset_name,
                "format": upload_request.format,
                "user_id": user_id,
                "manager_id": upload_request.manager_id
            }
            if upload_request.tags:
                default_tags.update(upload_request.tags)
            mlflow.set_tags(default_tags)
            
            # 통계 정보를 JSON으로 저장 및 업로드
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as stats_file:
                json.dump(statistics, stats_file, indent=2, default=str)
                stats_path = stats_file.name
            
            stats_size = os.path.getsize(stats_path)
            logger.info(f"통계 파일 생성: 크기 {stats_size} bytes")
            
            mlflow.log_artifact(stats_path, artifact_path=upload_request.artifact_path)
            logger.info("통계 아티팩트 업로드 완료")
            
            # 업로드 확인 (약간의 지연 후)
            import time
            time.sleep(1)  # S3 인덱싱 대기
            
            artifact_list = []
            try:
                uploaded_artifacts = client.list_artifacts(run.info.run_id, upload_request.artifact_path)
                artifact_list = [a.path for a in uploaded_artifacts]
                logger.info(f"업로드 확인: {len(artifact_list)}개 파일")
                
                for artifact in uploaded_artifacts:
                    logger.info(f"  - {artifact.path}: {artifact.file_size} bytes")
                    
            except Exception as e:
                logger.warning(f"아티팩트 확인 중 오류: {e}")
            
            run_info = {
                "run_id": run.info.run_id,
                "experiment_id": run.info.experiment_id,
                "artifact_uri": run.info.artifact_uri,
                "uploaded_artifacts": artifact_list
            }
            
            # 임시 파일 정리
            try:
                os.unlink(file_path)
                os.unlink(stats_path)
                logger.info("임시 파일 정리 완료")
            except Exception as cleanup_error:
                logger.warning(f"임시 파일 정리 중 오류: {cleanup_error}")

        logger.info(f"MLflow 업로드 완료: run_id={run.info.run_id}")

        return {
            "success": True,
            "manager_id": upload_request.manager_id,
            "message": f"데이터셋이 MLflow 실험 '{upload_request.experiment_name}'에 성공적으로 업로드되었습니다",
            "mlflow_info": {
                "experiment_name": upload_request.experiment_name,
                "dataset_name": upload_request.dataset_name,
                "format": upload_request.format,
                "artifact_path": upload_request.artifact_path,
                **run_info
            }
        }

    except ImportError:
        logger.error("MLflow 패키지가 설치되지 않음")
        raise HTTPException(status_code=500, detail="MLflow 패키지가 설치되지 않았습니다")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("MLflow 업로드 실패: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"MLflow 업로드 실패: {str(e)}")

@router.post("/mlflow/list-datasets",
    summary="MLflow 업로드된 데이터셋 목록 조회",
    description="MLflow에 업로드된 데이터셋들의 목록과 메타데이터를 조회합니다.",
    response_model=Dict[str, Any])
async def list_mlflow_datasets(request: Request, list_request: ListMlflowDatasetsRequest) -> Dict[str, Any]:
    """MLflow 데이터셋 목록 조회"""
    try:
        import mlflow
        import os
        from datetime import datetime
        
        user_id = extract_user_id_from_request(request)
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID가 제공되지 않았습니다")

        # S3/MinIO 설정
        os.environ['AWS_ACCESS_KEY_ID'] = 'minioadmin'
        os.environ['AWS_SECRET_ACCESS_KEY'] = 'minioadmin123'
        os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'https://minio.x2bee.com'
        
        # MLflow Tracking URI 설정
        if list_request.mlflow_tracking_uri:
            tracking_uri = list_request.mlflow_tracking_uri
        else:
            try:
                config_composer = get_config_composer(request)
                tracking_uri = config_composer.get_config_by_name("MLFLOW_TRACKING_URI").value
            except Exception as e:
                logger.warning("MLFLOW_TRACKING_URI 설정 가져오기 실패: %s", e)
                tracking_uri = "https://polar-mlflow-git.x2bee.com"
        
        mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.tracking.MlflowClient()
        
        logger.info(f"MLflow 데이터셋 목록 조회 시작 (user_id: {user_id})")
        
        # 실험 목록 가져오기
        if list_request.experiment_name:
            # 특정 실험만 조회
            try:
                experiment = client.get_experiment_by_name(list_request.experiment_name)
                experiments = [experiment] if experiment else []
            except Exception as e:
                logger.warning(f"실험 '{list_request.experiment_name}' 조회 실패: {e}")
                experiments = []
        else:
            # 모든 실험 조회
            experiments = client.search_experiments(max_results=1000)
        
        datasets = []
        
        for experiment in experiments:
            # 각 실험의 run들 조회
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                max_results=list_request.max_results,
                order_by=["start_time DESC"]
            )
            
            for run in runs:
                # dataset_name 태그가 있는 run만 필터링
                tags = run.data.tags
                if "dataset_name" not in tags:
                    continue
                
                # 파라미터와 메트릭 수집
                params = run.data.params
                metrics = run.data.metrics
                
                # 아티팩트 확인
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
                    logger.warning(f"Run {run.info.run_id} 아티팩트 조회 실패: {e}")
                
                # 데이터셋 정보 구성
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
                    "run_url": f"{tracking_uri}/#/experiments/{experiment.experiment_id}/runs/{run.info.run_id}"
                }
                
                # user_id로 필터링 (본인이 업로드한 데이터셋만 조회)
                if tags.get("user_id") == str(user_id):
                    datasets.append(dataset_info)
        
        logger.info(f"MLflow 데이터셋 목록 조회 완료: {len(datasets)}개 발견")
        
        return {
            "success": True,
            "message": f"{len(datasets)}개의 데이터셋을 찾았습니다",
            "total_count": len(datasets),
            "datasets": datasets,
            "filter": {
                "experiment_name": list_request.experiment_name,
                "user_id": user_id
            }
        }

    except ImportError:
        logger.error("MLflow 패키지가 설치되지 않음")
        raise HTTPException(status_code=500, detail="MLflow 패키지가 설치되지 않았습니다")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("MLflow 데이터셋 목록 조회 실패: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"데이터셋 목록 조회 실패: {str(e)}")

@router.post("/mlflow/get-dataset-columns",
    summary="MLflow 데이터셋의 컬럼 정보 조회",
    description="특정 run의 데이터셋 파일을 다운로드하여 컬럼 정보를 반환합니다.",
    response_model=Dict[str, Any])
async def get_mlflow_dataset_columns(request: Request, columns_request: GetMLflowDatasetColumnsRequest) -> Dict[str, Any]:
    """MLflow 데이터셋 컬럼 조회"""
    try:
        import mlflow
        import os
        import tempfile
        import pyarrow.parquet as pq
        import pyarrow.csv as csv
        
        user_id = extract_user_id_from_request(request)
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID가 제공되지 않았습니다")

        # S3/MinIO 설정
        os.environ['AWS_ACCESS_KEY_ID'] = 'minioadmin'
        os.environ['AWS_SECRET_ACCESS_KEY'] = 'minioadmin123'
        os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'https://minio.x2bee.com'
        
        # MLflow Tracking URI 설정
        if columns_request.mlflow_tracking_uri:
            tracking_uri = columns_request.mlflow_tracking_uri
        else:
            try:
                config_composer = get_config_composer(request)
                tracking_uri = config_composer.get_config_by_name("MLFLOW_TRACKING_URI").value
            except Exception as e:
                logger.warning("MLFLOW_TRACKING_URI 설정 가져오기 실패: %s", e)
                tracking_uri = "https://polar-mlflow-git.x2bee.com"
        
        mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.tracking.MlflowClient()
        
        logger.info(f"MLflow 데이터셋 컬럼 조회: run_id={columns_request.run_id}")
        
        # 아티팩트 목록 조회
        artifacts = client.list_artifacts(columns_request.run_id, columns_request.artifact_path)
        
        # 데이터셋 파일 찾기 (.csv 또는 .parquet)
        dataset_file = None
        for artifact in artifacts:
            if artifact.path.endswith('.csv') or artifact.path.endswith('.parquet'):
                dataset_file = artifact
                break
        
        if not dataset_file:
            raise HTTPException(
                status_code=404,
                detail=f"Run {columns_request.run_id}에서 데이터셋 파일(.csv 또는 .parquet)을 찾을 수 없습니다"
            )
        
        # 임시 디렉토리에 파일 다운로드
        with tempfile.TemporaryDirectory() as temp_dir:
            local_path = client.download_artifacts(
                columns_request.run_id,
                dataset_file.path,
                dst_path=temp_dir
            )
            
            # 파일 형식에 따라 컬럼 정보 추출
            columns_info = []
            file_format = 'parquet' if dataset_file.path.endswith('.parquet') else 'csv'
            
            try:
                if file_format == 'parquet':
                    table = pq.read_table(local_path)
                    schema = table.schema
                    
                    for field in schema:
                        columns_info.append({
                            "name": field.name,
                            "type": str(field.type),
                            "nullable": field.nullable
                        })
                    
                    num_rows = len(table)
                    
                else:  # CSV
                    # CSV 파일의 경우 처음 몇 줄만 읽어서 컬럼 정보 추출
                    table = csv.read_csv(local_path)
                    schema = table.schema
                    
                    for field in schema:
                        columns_info.append({
                            "name": field.name,
                            "type": str(field.type),
                            "nullable": field.nullable
                        })
                    
                    num_rows = len(table)
                
            except Exception as e:
                logger.error(f"파일 읽기 실패: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"데이터셋 파일 읽기 실패: {str(e)}"
                )
        
        logger.info(f"컬럼 정보 조회 완료: {len(columns_info)}개 컬럼")
        
        return {
            "success": True,
            "run_id": columns_request.run_id,
            "file_path": dataset_file.path,
            "file_format": file_format,
            "num_rows": num_rows,
            "num_columns": len(columns_info),
            "columns": columns_info
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("MLflow 데이터셋 컬럼 조회 실패: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"컬럼 조회 실패: {str(e)}")
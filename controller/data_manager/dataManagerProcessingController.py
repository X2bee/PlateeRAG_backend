from fastapi import APIRouter, HTTPException, Request, File, UploadFile, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import logging
import os
from typing import Dict, Any, Optional, List
from datetime import datetime
from controller.helper.singletonHelper import get_db_manager, get_data_manager_registry, get_config_composer
from controller.helper.controllerHelper import extract_user_id_from_request

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

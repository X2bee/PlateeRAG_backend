from fastapi import APIRouter, HTTPException, Request, File, UploadFile, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import logging
import os
from typing import Dict, Any, Optional, List
from datetime import datetime
from controller.helper.singletonHelper import get_db_manager, get_data_manager_registry
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

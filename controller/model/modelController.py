"""Endpoints for managing uploaded machine learning models."""
from __future__ import annotations
import hashlib
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, List

import aiofiles
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status
from pydantic import BaseModel, Field, field_validator, model_validator

from controller.helper.controllerHelper import extract_user_id_from_request
from controller.helper.singletonHelper import get_db_manager, get_config_composer
from service.database.logger_helper import create_logger
from service.model.model_registry_service import ModelRegistryService
from service.model.model_inference_service import ModelInferenceService

logger = logging.getLogger("model-controller")

router = APIRouter(prefix="/api/models", tags=["model-registry"])

ALLOWED_EXTENSIONS = {".pkl", ".joblib"}
DEFAULT_FRAMEWORK = "sklearn"
_inference_service = ModelInferenceService()

class InferenceRequest(BaseModel):
    model_id: Optional[int] = Field(default=None, description="등록된 모델의 ID")
    model_name: Optional[str] = Field(default=None, description="등록된 모델 이름")
    model_version: Optional[str] = Field(default=None, description="모델 버전")
    inputs: List[Any] = Field(..., description="모델에 전달할 입력 데이터. 리스트 또는 리스트의 리스트/딕셔너리")
    return_probabilities: bool = Field(default=False, description="분류 모델의 확률 반환 여부")

    @field_validator("inputs")
    @classmethod
    def validate_inputs(cls, value: List[Any]) -> List[Any]:
        if not isinstance(value, list) or not value:
            raise ValueError("inputs must be a non-empty list")
        return value

    @model_validator(mode="after")
    def ensure_identifier(self) -> "InferenceRequest":
        if self.model_id is None and not self.model_name:
            raise ValueError("Either model_id or model_name must be provided")
        return self


def _sanitize_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]", "_", value).strip("_.")
    return cleaned or "model"


def _ensure_model_directory(request: Request) -> Path:
    config_composer = get_config_composer(request)
    try:
        storage_config = config_composer.get_config_by_name("MODEL_STORAGE_DIRECTORY")
        base_dir = storage_config.value
    except KeyError:
        base_dir = "models"

    base_path = Path(base_dir)
    if not base_path.is_absolute():
        base_path = Path.cwd() / base_path
    base_path.mkdir(parents=True, exist_ok=True)
    return base_path


def _parse_optional_json(raw_value: Optional[str], field_name: str) -> Optional[Any]:
    if raw_value is None or raw_value == "":
        return None
    try:
        return json.loads(raw_value)
    except json.JSONDecodeError as exc:  # type: ignore[assignment]
        if field_name in {"input_schema", "output_schema"}:
            return [item.strip() for item in raw_value.split(",") if item.strip()]
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"INVALID_{field_name.upper()}"
        ) from exc


def _coerce_user_id(user_id: Any) -> Optional[int]:
    try:
        return int(user_id) if user_id is not None else None
    except (TypeError, ValueError):
        return None


@router.post("/upload")
async def upload_model(
    request: Request,
    file: UploadFile = File(...),
    model_name: str = Form(...),
    model_version: Optional[str] = Form(None),
    framework: str = Form(DEFAULT_FRAMEWORK),
    description: Optional[str] = Form(None),
    input_schema: Optional[str] = Form(None),
    output_schema: Optional[str] = Form(None),
    metadata: Optional[str] = Form(None),
):
    user_id = extract_user_id_from_request(request)
    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, _coerce_user_id(user_id), request)

    framework_normalized = framework.lower().strip() if framework else DEFAULT_FRAMEWORK
    if framework_normalized not in {DEFAULT_FRAMEWORK}:
        backend_log.warn(
            "Unsupported framework requested",
            metadata={"framework": framework}
        )
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="UNSUPPORTED_FRAMEWORK")

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        backend_log.warn(
            "Rejected model upload due to unsupported extension",
            metadata={"extension": suffix}
        )
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="UNSUPPORTED_MODEL_EXTENSION")

    registry_service = ModelRegistryService(app_db)
    existing_model = registry_service.get_model_by_name(model_name, model_version)
    if existing_model:
        backend_log.warn(
            "Duplicate model registration attempt",
            metadata={"model_name": model_name, "model_version": model_version}
        )
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="MODEL_ALREADY_REGISTERED")

    storage_root = _ensure_model_directory(request)
    safe_model_name = _sanitize_name(model_name)
    target_dir = storage_root / safe_model_name
    if model_version:
        target_dir = target_dir / _sanitize_name(model_version)
    target_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    artifact_name_parts = [safe_model_name]
    if model_version:
        artifact_name_parts.append(_sanitize_name(model_version))
    artifact_name_parts.append(timestamp)
    artifact_filename = "__".join(artifact_name_parts) + suffix
    artifact_path = target_dir / artifact_filename

    hasher = hashlib.sha256()
    total_bytes = 0

    try:
        async with aiofiles.open(artifact_path, "wb") as buffer:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total_bytes += len(chunk)
                hasher.update(chunk)
                await buffer.write(chunk)
    except Exception as exc:  # pragma: no cover - IO failure path
        if artifact_path.exists():
            artifact_path.unlink(missing_ok=True)
        backend_log.error("Model upload failed during write", exception=exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="MODEL_UPLOAD_FAILED") from exc
    finally:
        await file.close()

    checksum = hasher.hexdigest()

    # Prevent identical artifacts from being registered twice
    duplicate_artifact = registry_service.get_model_by_checksum(checksum)
    if duplicate_artifact:
        artifact_path.unlink(missing_ok=True)
        backend_log.warn(
            "Duplicate artifact detected via checksum",
            metadata={
                "model_name": model_name,
                "checksum": checksum,
                "existing_model_id": duplicate_artifact.id,
            }
        )
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="MODEL_ARTIFACT_ALREADY_EXISTS")

    parsed_metadata = _parse_optional_json(metadata, "metadata") or {}
    parsed_input_schema = _parse_optional_json(input_schema, "input_schema")
    parsed_output_schema = _parse_optional_json(output_schema, "output_schema")

    model_payload: Dict[str, Any] = {
        "model_name": model_name,
        "model_version": model_version,
        "framework": framework_normalized,
        "description": description,
        "file_path": str(artifact_path.resolve()),
        "file_size": total_bytes,
        "file_checksum": checksum,
        "input_schema": parsed_input_schema,
        "output_schema": parsed_output_schema,
        "metadata": parsed_metadata,
        "uploaded_by": _coerce_user_id(user_id),
        "status": "active",
    }

    try:
        created_model = registry_service.register_model(model_payload)
    except Exception as exc:
        if artifact_path.exists():
            artifact_path.unlink(missing_ok=True)
        backend_log.error("Failed to register model metadata", exception=exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="MODEL_REGISTRATION_FAILED") from exc

    backend_log.success(
        "Model uploaded successfully",
        metadata={
            "model_id": created_model.id,
            "model_name": created_model.model_name,
            "model_version": created_model.model_version,
            "file_path": created_model.file_path,
            "file_size": created_model.file_size,
        }
    )

    return {
        "model_id": created_model.id,
        "model_name": created_model.model_name,
        "model_version": created_model.model_version,
        "framework": created_model.framework,
        "file_path": created_model.file_path,
        "file_checksum": created_model.file_checksum,
        "file_size": created_model.file_size,
        "input_schema": created_model.input_schema,
        "output_schema": created_model.output_schema,
        "metadata": created_model.metadata,
    }


@router.post("/infer")
async def run_inference(request: Request, payload: InferenceRequest):
    user_id = extract_user_id_from_request(request)
    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, _coerce_user_id(user_id), request)
    registry_service = ModelRegistryService(app_db)

    model_record = None
    if payload.model_id is not None:
        model_record = registry_service.get_model_by_id(payload.model_id)
    else:
        model_record = registry_service.get_model_by_name(payload.model_name, payload.model_version)

    if not model_record:
        backend_log.warn(
            "Inference requested for unknown model",
            metadata={"model_id": payload.model_id, "model_name": payload.model_name}
        )
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="MODEL_NOT_FOUND")

    try:
        inference_result = _inference_service.predict(
            file_path=model_record.file_path,
            inputs=payload.inputs,
            input_schema=model_record.get_input_schema(),
            return_probabilities=payload.return_probabilities,
        )
    except FileNotFoundError as exc:
        backend_log.error("Stored model artifact missing", exception=exc)
        raise HTTPException(status_code=status.HTTP_410_GONE, detail="MODEL_ARTIFACT_MISSING") from exc
    except ValueError as exc:
        backend_log.warn("Invalid inference request", metadata={"error": str(exc)})
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - model specific failure path
        backend_log.error("Inference execution failed", exception=exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="MODEL_INFERENCE_FAILED") from exc

    backend_log.success(
        "Inference executed",
        metadata={
            "model_id": model_record.id,
            "model_name": model_record.model_name,
            "prediction_count": len(inference_result.get("predictions", []))
        }
    )

    return {
        "model": {
            "id": model_record.id,
            "name": model_record.model_name,
            "version": model_record.model_version,
            "framework": model_record.framework,
        },
        "result": inference_result,
    }

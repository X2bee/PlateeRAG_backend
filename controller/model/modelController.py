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
from controller.helper.singletonHelper import get_db_manager, get_config_composer, get_mlflow_service
from service.database.logger_helper import create_logger
from service.model.model_registry_service import ModelRegistryService
from service.model.model_inference_service import ModelInferenceService
from service.model.model_deletion_service import ModelDeletionService
from service.database.models.backend import BackendLogs

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


def _parse_mlflow_uri(uri: Optional[str]) -> Optional[tuple[str, str]]:
    if not isinstance(uri, str):
        return None
    prefix = "mlflow://"
    if not uri.startswith(prefix):
        return None
    remainder = uri[len(prefix):]
    if "/" in remainder:
        run_id, artifact_path = remainder.split("/", 1)
    else:
        run_id, artifact_path = remainder, ""
    return run_id, artifact_path


def _resolve_mlflow_binding(
    model_metadata: Optional[Dict[str, Any]],
    file_path: Optional[str]
) -> tuple[Optional[str], Optional[str]]:
    run_id = None
    artifact_path = None

    if isinstance(model_metadata, dict):
        run_id = model_metadata.get("mlflow_run_id")
        artifact_path = model_metadata.get("mlflow_artifact_path")

    if run_id and artifact_path:
        return str(run_id), str(artifact_path)

    parsed = _parse_mlflow_uri(file_path)
    if parsed:
        return parsed

    return None, None


def _serialize_backend_log(log: BackendLogs) -> Dict[str, Any]:
    payload = log.to_dict()
    metadata = payload.get("metadata")

    if isinstance(metadata, str) and metadata:
        try:
            payload["metadata"] = json.loads(metadata)
        except json.JSONDecodeError:
            payload["metadata"] = metadata
    return payload


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
    mlflow_run_id: Optional[str] = Form(None),
    mlflow_experiment_id: Optional[str] = Form(None),
    mlflow_run_name: Optional[str] = Form(None),
    mlflow_artifact_path: Optional[str] = Form(None),
):
    user_id = extract_user_id_from_request(request)
    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, _coerce_user_id(user_id), request)

    logger.info(
        "[upload_model] Upload requested | user_id=%s model_name=%s version=%s filename=%s",
        user_id,
        model_name,
        model_version,
        file.filename,
    )

    try:
        mlflow_service = get_mlflow_service(request)
    except HTTPException as exc:
        backend_log.error("MLflow service unavailable", exception=exc)
        raise

    framework_normalized = framework.lower().strip() if framework else DEFAULT_FRAMEWORK
    if framework_normalized not in {DEFAULT_FRAMEWORK}:
        logger.warning(
            "[upload_model] Unsupported framework | user_id=%s framework=%s",
            user_id,
            framework,
        )
        backend_log.warn(
            "Unsupported framework requested",
            metadata={"framework": framework}
        )
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="UNSUPPORTED_FRAMEWORK")

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        logger.warning(
            "[upload_model] Unsupported extension | user_id=%s extension=%s",
            user_id,
            suffix,
        )
        backend_log.warn(
            "Rejected model upload due to unsupported extension",
            metadata={"extension": suffix}
        )
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="UNSUPPORTED_MODEL_EXTENSION")

    registry_service = ModelRegistryService(app_db)
    existing_model = registry_service.get_model_by_name(model_name, model_version)
    if existing_model:
        logger.warning(
            "[upload_model] Duplicate model detected | user_id=%s model_name=%s version=%s",
            user_id,
            model_name,
            model_version,
        )
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
        logger.exception(
            "[upload_model] Failed to persist artifact | user_id=%s path=%s",
            user_id,
            artifact_path,
        )
        if artifact_path.exists():
            artifact_path.unlink(missing_ok=True)
        backend_log.error("Model upload failed during write", exception=exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="MODEL_UPLOAD_FAILED") from exc
    finally:
        await file.close()

    checksum = hasher.hexdigest()
    logger.info(
        "[upload_model] Upload finished writing | user_id=%s bytes=%s checksum=%s",
        user_id,
        total_bytes,
        checksum,
    )

    duplicate_artifact = registry_service.get_model_by_checksum(checksum)
    if duplicate_artifact:
        logger.warning(
            "[upload_model] Duplicate artifact checksum detected | user_id=%s checksum=%s existing_id=%s",
            user_id,
            checksum,
            duplicate_artifact.id,
        )
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

    artifact_directory = (mlflow_artifact_path or safe_model_name or "artifacts").strip("/")
    run_name_for_mlflow = mlflow_run_name or model_name

    try:
        resolved_run_id = mlflow_service.ensure_run(
            run_id=mlflow_run_id,
            experiment_id=mlflow_experiment_id,
            run_name=run_name_for_mlflow,
        )
    except ValueError as run_error:
        logger.error(
            "[upload_model] Invalid MLflow run parameters | user_id=%s error=%s",
            user_id,
            run_error,
        )
        backend_log.error("Invalid MLflow run parameters", exception=run_error)
        artifact_path.unlink(missing_ok=True)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(run_error)) from run_error

    resolved_experiment_id = mlflow_experiment_id or mlflow_service.default_experiment_id

    try:
        artifact_relative_path = mlflow_service.log_artifact(
            resolved_run_id,
            artifact_path,
            artifact_dir=artifact_directory,
        )
        try:
            run_info = mlflow_service.client.get_run(resolved_run_id)
            resolved_experiment_id = getattr(run_info.info, "experiment_id", resolved_experiment_id)
        except Exception:  # pragma: no cover
            pass
        artifact_uri = mlflow_service.build_artifact_uri(resolved_run_id, artifact_relative_path)
    except Exception as mlflow_exc:
        logger.exception(
            "[upload_model] Failed to store artifact in MLflow | user_id=%s run_id=%s",
            user_id,
            resolved_run_id,
        )
        backend_log.error("Failed to upload artifact to MLflow", exception=mlflow_exc)
        artifact_path.unlink(missing_ok=True)
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, detail="MLFLOW_UPLOAD_FAILED") from mlflow_exc
    finally:
        artifact_path.unlink(missing_ok=True)

    parsed_metadata.update(
        {
            "mlflow_run_id": resolved_run_id,
            "mlflow_artifact_path": artifact_relative_path,
            "mlflow_artifact_uri": artifact_uri,
            "mlflow_tracking_uri": mlflow_service.tracking_uri,
        }
    )
    if resolved_experiment_id:
        parsed_metadata.setdefault("mlflow_experiment_id", resolved_experiment_id)
    if run_name_for_mlflow:
        parsed_metadata.setdefault("mlflow_run_name", run_name_for_mlflow)

    model_payload: Dict[str, Any] = {
        "model_name": model_name,
        "model_version": model_version,
        "framework": framework_normalized,
        "description": description,
        "file_path": artifact_uri,
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
        logger.exception(
            "[upload_model] Failed to register model metadata | user_id=%s model_name=%s",
            user_id,
            model_name,
        )
        backend_log.error("Failed to register model metadata", exception=exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="MODEL_REGISTRATION_FAILED") from exc

    logger.info(
        "[upload_model] Model registered | user_id=%s model_id=%s",
        user_id,
        created_model.id,
    )
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


@router.get("/")
async def list_models(request: Request, limit: int = 100, offset: int = 0):
    user_id = extract_user_id_from_request(request)
    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, _coerce_user_id(user_id), request)

    logger.info(
        "[list_models] Listing models | user_id=%s limit=%s offset=%s",
        user_id,
        limit,
        offset,
    )
    registry_service = ModelRegistryService(app_db)
    models = registry_service.list_models(limit=limit, offset=offset)

    serialized = [
        {
            "id": model.id,
            "model_name": model.model_name,
            "model_version": model.model_version,
            "framework": model.framework,
            "file_path": model.file_path,
            "file_size": model.file_size,
            "file_checksum": model.file_checksum,
            "status": model.status,
            "uploaded_by": model.uploaded_by,
            "created_at": model.created_at.isoformat() if model.created_at else None,
            "updated_at": model.updated_at.isoformat() if model.updated_at else None,
        }
        for model in models
    ]

    logger.info(
        "[list_models] Retrieved %s models | user_id=%s",
        len(serialized),
        user_id,
    )
    backend_log.success(
        "Listed models",
        metadata={"count": len(serialized), "limit": limit, "offset": offset}
    )

    return {"items": serialized, "limit": limit, "offset": offset}


@router.get("/logs")
async def list_model_logs(
    request: Request,
    limit: int = 250,
    offset: int = 0,
    log_level: Optional[str] = None,
    message: Optional[str] = None,
    function_name: Optional[str] = None,
    api_endpoint: Optional[str] = None,
):
    user_id = extract_user_id_from_request(request)
    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, _coerce_user_id(user_id), request)

    logger.info(
        "[list_model_logs] Fetching backend logs | user_id=%s limit=%s offset=%s level=%s function=%s endpoint=%s message=%s",
        user_id,
        limit,
        offset,
        log_level,
        function_name,
        api_endpoint,
        message,
    )

    if app_db is None:
        backend_log.error("Database manager not available for log retrieval")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="DATABASE_NOT_AVAILABLE")

    coerced_user_id = _coerce_user_id(user_id)
    if coerced_user_id is None:
        logger.warning(
            "[list_model_logs] Missing user context | request_path=%s",
            request.url.path,
        )
        backend_log.warn("Log retrieval attempted without valid user context")
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="USER_NOT_IDENTIFIED")

    limit = max(1, min(limit, 500))
    offset = max(0, offset)

    conditions: Dict[str, Any] = {"user_id": coerced_user_id}

    if log_level:
        conditions["log_level"] = log_level.upper()
    if function_name:
        conditions["function_name"] = function_name
    if api_endpoint:
        conditions["api_endpoint"] = api_endpoint
    if message:
        conditions["message__like__"] = message

    try:
        logs = app_db.find_by_condition(
            BackendLogs,
            conditions,
            limit=limit,
            offset=offset,
            orderby="created_at",
            orderby_asc=False,
        )
    except Exception as exc:
        logger.exception(
            "[list_model_logs] Failed to fetch logs | user_id=%s",
            user_id,
        )
        backend_log.error("Failed to fetch backend logs", exception=exc)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="LOG_FETCH_FAILED") from exc

    serialized_logs = [_serialize_backend_log(entry) for entry in logs]

    logger.info(
        "[list_model_logs] Retrieved %s log entries | user_id=%s",
        len(serialized_logs),
        user_id,
    )
    backend_log.success(
        "Retrieved backend logs",
        metadata={
            "returned": len(serialized_logs),
            "limit": limit,
            "offset": offset,
            "level": log_level,
            "function_name": function_name,
            "api_endpoint": api_endpoint,
            "message_filter": message,
        },
    )

    return {
        "logs": serialized_logs,
        "pagination": {
            "limit": limit,
            "offset": offset,
            "returned": len(serialized_logs),
        },
        "filters": {
            "log_level": log_level.upper() if log_level else None,
            "function_name": function_name,
            "api_endpoint": api_endpoint,
            "message": message,
        },
    }


@router.get("/{model_id}")
async def get_model_detail(request: Request, model_id: int):
    user_id = extract_user_id_from_request(request)
    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, _coerce_user_id(user_id), request)

    logger.info(
        "[get_model_detail] Fetching model detail | user_id=%s model_id=%s",
        user_id,
        model_id,
    )
    registry_service = ModelRegistryService(app_db)
    model = registry_service.get_model_by_id(model_id)

    if not model:
        logger.warning(
            "[get_model_detail] Model not found | user_id=%s model_id=%s",
            user_id,
            model_id,
        )
        backend_log.warn(
            "Requested missing model",
            metadata={"model_id": model_id}
        )
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="MODEL_NOT_FOUND")

    logger.info(
        "[get_model_detail] Model retrieved | user_id=%s model_id=%s",
        user_id,
        model_id,
    )
    backend_log.success(
        "Retrieved model detail",
        metadata={"model_id": model_id}
    )

    return {
        "id": model.id,
        "model_name": model.model_name,
        "model_version": model.model_version,
        "framework": model.framework,
        "file_path": model.file_path,
        "file_size": model.file_size,
        "file_checksum": model.file_checksum,
        "input_schema": model.input_schema,
        "output_schema": model.output_schema,
        "metadata": model.metadata,
        "status": model.status,
        "uploaded_by": model.uploaded_by,
        "created_at": model.created_at.isoformat() if model.created_at else None,
        "updated_at": model.updated_at.isoformat() if model.updated_at else None,
    }


@router.post("/infer")
async def run_inference(request: Request, payload: InferenceRequest):
    user_id = extract_user_id_from_request(request)
    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, _coerce_user_id(user_id), request)
    registry_service = ModelRegistryService(app_db)

    logger.info(
        "[run_inference] Inference requested | user_id=%s model_id=%s model_name=%s version=%s",
        user_id,
        payload.model_id,
        payload.model_name,
        payload.model_version,
    )

    model_record = None
    if payload.model_id is not None:
        model_record = registry_service.get_model_by_id(payload.model_id)
    else:
        model_record = registry_service.get_model_by_name(payload.model_name, payload.model_version)

    if not model_record:
        logger.warning(
            "[run_inference] Model not found for inference | user_id=%s model_id=%s model_name=%s",
            user_id,
            payload.model_id,
            payload.model_name,
        )
        backend_log.warn(
            "Inference requested for unknown model",
            metadata={"model_id": payload.model_id, "model_name": payload.model_name}
        )
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="MODEL_NOT_FOUND")

    local_model_path = model_record.file_path
    run_id, artifact_path = _resolve_mlflow_binding(model_record.metadata, model_record.file_path)
    if run_id and artifact_path:
        try:
            mlflow_service = get_mlflow_service(request)
        except HTTPException as exc:
            backend_log.error("MLflow service unavailable", exception=exc)
            raise

        artifact_uri = mlflow_service.build_artifact_uri(run_id, artifact_path)
        try:
            cached_path = mlflow_service.ensure_local_artifact(
                artifact_uri,
                expected_checksum=model_record.file_checksum,
            )
            local_model_path = str(cached_path)
            logger.info(
                "[run_inference] Resolved MLflow artifact | user_id=%s run_id=%s artifact_path=%s local_path=%s",
                user_id,
                run_id,
                artifact_path,
                local_model_path,
            )
        except Exception as exc:
            logger.exception(
                "[run_inference] Failed to download MLflow artifact | user_id=%s run_id=%s artifact_path=%s",
                user_id,
                run_id,
                artifact_path,
            )
            backend_log.error("Failed to download MLflow artifact", exception=exc)
            raise HTTPException(status.HTTP_502_BAD_GATEWAY, detail="MLFLOW_ARTIFACT_UNAVAILABLE") from exc

    try:
        inference_result = _inference_service.predict(
            file_path=local_model_path,
            inputs=payload.inputs,
            input_schema=model_record.get_input_schema(),
            return_probabilities=payload.return_probabilities,
        )
    except FileNotFoundError as exc:
        logger.exception(
            "[run_inference] Artifact missing | user_id=%s model_id=%s path=%s",
            user_id,
            model_record.id if model_record else None,
            model_record.file_path if model_record else None,
        )
        backend_log.error("Stored model artifact missing", exception=exc)
        raise HTTPException(status_code=status.HTTP_410_GONE, detail="MODEL_ARTIFACT_MISSING") from exc
    except ValueError as exc:
        logger.warning(
            "[run_inference] Invalid inference request | user_id=%s error=%s",
            user_id,
            exc,
        )
        backend_log.warn("Invalid inference request", metadata={"error": str(exc)})
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - model specific failure path
        logger.exception(
            "[run_inference] Inference failure | user_id=%s model_id=%s",
            user_id,
            model_record.id if model_record else None,
        )
        backend_log.error("Inference execution failed", exception=exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="MODEL_INFERENCE_FAILED") from exc

    logger.info(
        "[run_inference] Inference completed | user_id=%s model_id=%s predictions=%s",
        user_id,
        model_record.id,
        len(inference_result.get("predictions", [])),
    )
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


@router.post("/sync")
async def sync_model_artifacts(request: Request):
    user_id = extract_user_id_from_request(request)
    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, _coerce_user_id(user_id), request)

    logger.info(
        "[sync_model_artifacts] Sync requested | user_id=%s",
        user_id,
    )
    registry_service = ModelRegistryService(app_db)
    deletion_service = ModelDeletionService(app_db)

    models = registry_service.list_models(limit=10_000)
    removed_ids = []

    try:
        mlflow_service = get_mlflow_service(request)
    except HTTPException:
        mlflow_service = None

    for model in models:
        run_id, artifact_path = _resolve_mlflow_binding(model.metadata, model.file_path)
        if run_id and artifact_path:
            if not mlflow_service:
                logger.warning(
                    "[sync_model_artifacts] MLflow service unavailable; skipping remote check | model_id=%s",
                    model.id,
                )
                continue
            if not mlflow_service.artifact_exists(run_id, artifact_path):
                logger.warning(
                    "[sync_model_artifacts] Missing MLflow artifact detected | model_id=%s run_id=%s path=%s",
                    model.id,
                    run_id,
                    artifact_path,
                )
                if deletion_service.delete(model.id, model=model, remove_artifact=False):
                    removed_ids.append(model.id)
            continue

        try:
            path = Path(model.file_path)
        except (TypeError, ValueError):
            path = None

        if not path or not path.exists():
            if deletion_service.delete(model.id, model=model, remove_artifact=False):
                removed_ids.append(model.id)

    logger.info(
        "[sync_model_artifacts] Sync complete | user_id=%s checked=%s removed=%s",
        user_id,
        len(models),
        len(removed_ids),
    )
    backend_log.success(
        "Model artifact sync completed",
        metadata={"checked": len(models), "removed": len(removed_ids)}
    )

    return {
        "checked": len(models),
        "removed": len(removed_ids),
        "removed_ids": removed_ids,
    }


@router.delete("/{model_id}")
async def delete_model(request: Request, model_id: int):
    user_id = extract_user_id_from_request(request)
    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, _coerce_user_id(user_id), request)

    logger.info(
        "[delete_model] Delete requested | user_id=%s model_id=%s",
        user_id,
        model_id,
    )
    deletion_service = ModelDeletionService(app_db)
    existing = deletion_service.get_by_id(model_id)

    if not existing:
        logger.warning(
            "[delete_model] Model not found | user_id=%s model_id=%s",
            user_id,
            model_id,
        )
        backend_log.warn(
            "Attempted to delete missing model",
            metadata={"model_id": model_id}
        )
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="MODEL_NOT_FOUND")

    remove_local_artifact = True
    run_id, artifact_path = _resolve_mlflow_binding(existing.metadata, existing.file_path)
    if run_id and artifact_path:
        try:
            mlflow_service = get_mlflow_service(request)
        except HTTPException as exc:
            backend_log.error("MLflow service unavailable", exception=exc)
            raise

        try:
            mlflow_service.delete_artifact(run_id, artifact_path)
            logger.info(
                "[delete_model] Deleted MLflow artifact | user_id=%s model_id=%s run_id=%s path=%s",
                user_id,
                model_id,
                run_id,
                artifact_path,
            )
        except Exception as exc:
            logger.exception(
                "[delete_model] Failed to delete MLflow artifact | user_id=%s model_id=%s run_id=%s path=%s",
                user_id,
                model_id,
                run_id,
                artifact_path,
            )
            backend_log.error("Failed to delete MLflow artifact", exception=exc)
            raise HTTPException(status.HTTP_502_BAD_GATEWAY, detail="MLFLOW_DELETE_FAILED") from exc

        remove_local_artifact = False

    try:
        deleted = deletion_service.delete(
            model_id,
            model=existing,
            remove_artifact=remove_local_artifact,
        )
    except RuntimeError as exc:
        logger.exception(
            "[delete_model] Failed to delete artifact | user_id=%s model_id=%s",
            user_id,
            model_id,
        )
        backend_log.error(
            "Failed to remove model artifact",
            metadata={"model_id": model_id},
            exception=exc,
        )
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="MODEL_FILE_DELETE_FAILED") from exc

    if deleted:
        logger.info(
            "[delete_model] Model deleted | user_id=%s model_id=%s",
            user_id,
            model_id,
        )
        backend_log.success(
            "Model deleted",
            metadata={
                "model_id": model_id,
                "model_name": existing.model_name,
                "model_version": existing.model_version,
            }
        )
        return {"message": "MODEL_DELETED", "model_id": model_id}

    logger.error(
        "[delete_model] Delete failed | user_id=%s model_id=%s",
        user_id,
        model_id,
    )
    backend_log.error(
        "Failed to delete model",
        metadata={"model_id": model_id}
    )
    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="MODEL_DELETE_FAILED")

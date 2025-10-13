"""Endpoints for managing uploaded machine learning models."""
from __future__ import annotations
import ast
import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, List

import tempfile
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status
from pydantic import BaseModel, Field, field_validator, model_validator

from controller.helper.controllerHelper import extract_user_id_from_request
from controller.helper.singletonHelper import get_db_manager, get_mlflow_service
from service.database.logger_helper import create_logger
from service.model.model_registry_service import ModelRegistryService
from service.model.model_inference_service import ModelInferenceService
from service.model.model_deletion_service import ModelDeletionService
from service.database.models.backend import BackendLogs
from service.model.mlflow_utils import extract_mlflow_info

logger = logging.getLogger("model-controller")

router = APIRouter(prefix="/api/models", tags=["model-registry"])

ALLOWED_EXTENSIONS = {".pkl", ".joblib"}
DEFAULT_FRAMEWORK = "sklearn"
_inference_service = ModelInferenceService()

_MLFLOW_STAGE_MAP = {
    "none": "None",
    "staging": "Staging",
    "production": "Production",
    "archived": "Archived",
}

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


class ModelStageUpdateRequest(BaseModel):
    stage: str = Field(..., description="변경할 MLflow 모델 Stage (None, Staging, Production, Archived)")
    archive_existing_versions: bool = Field(
        default=False,
        description="Production 단계로 변경 시 기존 Production 버전을 Archived 상태로 전환할지 여부",
    )

    @field_validator("stage")
    @classmethod
    def validate_stage(cls, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("Stage must be a string")
        normalized = value.strip().lower()
        stage = _MLFLOW_STAGE_MAP.get(normalized)
        if not stage:
            raise ValueError("Stage must be one of: None, Staging, Production, Archived")
        return stage

def _sanitize_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]", "_", value).strip("_.")
    return cleaned or "model"



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

    storage_location = None

    if isinstance(model_metadata, dict):
        run_id = model_metadata.get("mlflow_run_id")
        artifact_path = model_metadata.get("mlflow_artifact_path")
        artifact_uri = model_metadata.get("mlflow_artifact_uri")
        storage_location = model_metadata.get("mlflow_storage_location")

        mlflow_block = model_metadata.get("mlflow")
        if isinstance(mlflow_block, dict):
            run_id = run_id or mlflow_block.get("run_id")
            artifact_path = artifact_path or mlflow_block.get("artifact_path")
            artifact_uri = artifact_uri or mlflow_block.get("artifact_uri")
            storage_location = storage_location or mlflow_block.get("storage_location")

            additional = mlflow_block.get("additional_metadata")
            if isinstance(additional, dict):
                run_id = run_id or additional.get("run_id")
                artifact_path = artifact_path or additional.get("artifact_path")
                artifact_uri = artifact_uri or additional.get("artifact_uri")
                storage_location = storage_location or additional.get("storage_location")

        if artifact_uri and not artifact_path:
            parsed_run, parsed_path = _parse_registered_model_source(str(artifact_uri), run_id=run_id)
            if parsed_run:
                run_id = run_id or parsed_run
            if parsed_path:
                artifact_path = parsed_path

        if storage_location and not artifact_path:
            parsed_run, parsed_path = _parse_registered_model_source(str(storage_location), run_id=run_id)
            if parsed_run:
                run_id = run_id or parsed_run
            if parsed_path:
                artifact_path = parsed_path

    if run_id and artifact_path:
        return str(run_id), str(artifact_path)

    parsed = _parse_mlflow_uri(file_path)
    if parsed:
        return parsed

    return None, None


def _get_mlflow_service_or_none(request: Request):
    try:
        return get_mlflow_service(request)
    except HTTPException:
        return None


def _parse_possible_json(value: Any) -> Any:
    if isinstance(value, str):
        candidate = value.strip()
        if not candidate:
            return value
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return value
    return value


def _format_timestamp_ms(timestamp_ms: Any) -> Optional[str]:
    if isinstance(timestamp_ms, (int, float)):
        try:
            return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).isoformat()
        except (OverflowError, OSError, ValueError):  # pragma: no cover - defensive guard
            return None
    return None


def _parse_registered_model_source(
    source: Optional[str],
    *,
    run_id: Optional[str] = None,
) -> tuple[Optional[str], Optional[str]]:
    """Extract run identifier and artifact path from an MLflow model version source."""

    if not source:
        return run_id, None

    parsed_run_id = run_id
    artifact_path: Optional[str] = None

    normalized = source.strip()
    remainder = normalized

    if normalized.startswith("runs:/"):
        remainder = normalized[len("runs:/"):]
    elif normalized.startswith("mlflow-artifacts:/"):
        remainder = normalized[len("mlflow-artifacts:/"):]
    elif "://" in normalized:
        _, _, remainder = normalized.partition("://")

    remainder = remainder.lstrip("/")

    if "/artifacts/" in remainder:
        run_segment, artifact_segment = remainder.split("/artifacts/", 1)
        run_segment = run_segment.rstrip("/")
        last_component = run_segment.split("/")[-1] if run_segment else None
        if last_component:
            parsed_run_id = parsed_run_id or last_component
        artifact_path = artifact_segment.strip("/")
    elif "/" in remainder:
        possible_run_id, artifact_segment = remainder.split("/", 1)
        parsed_run_id = parsed_run_id or possible_run_id.strip("/") or None
        artifact_path = artifact_segment.strip("/")
    else:
        parsed_run_id = parsed_run_id or remainder.strip("/") or None
        artifact_path = ""

    return parsed_run_id, artifact_path


def _parse_literal_list(value: Any) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(stripped)
            except (ValueError, SyntaxError):
                cleaned = stripped.strip("[](){}")
                if "," in cleaned:
                    return [segment.strip().strip('"\'') for segment in cleaned.split(",") if segment.strip()]
                return [cleaned.strip('"\'')]
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    return None


def _collect_schema_column_names(schema_payload: Any) -> List[str]:
    names: List[str] = []

    if isinstance(schema_payload, dict):
        name_value = schema_payload.get("name")
        if isinstance(name_value, str) and name_value.strip():
            names.append(name_value.strip())

        column_names = schema_payload.get("column_names")
        if isinstance(column_names, list):
            names.extend(str(item) for item in column_names if isinstance(item, (str, int, float)))

        keys_to_traverse = {
            "mlflow_colspec",
            "colspec",
            "columns",
            "fields",
            "items",
            "inputs",
            "outputs",
            "elements",
            "properties",
        }

        for key, value in schema_payload.items():
            if key in keys_to_traverse or key.endswith("_schema") or key in {"schema", "signature"}:
                names.extend(_collect_schema_column_names(value))

    elif isinstance(schema_payload, list):
        for item in schema_payload:
            names.extend(_collect_schema_column_names(item))

    return names


def _deduplicate_names(candidates: List[str]) -> Optional[List[str]]:
    ordered: List[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = str(candidate).strip()
        if not normalized:
            continue
        if normalized not in seen:
            seen.add(normalized)
            ordered.append(normalized)
    return ordered or None


def _resolve_schema_download_params(
    schema_uri: Optional[Any],
    *,
    fallback_run_id: Optional[str],
    mlflow_service: Any,
) -> tuple[Optional[str], Optional[str]]:
    if schema_uri is None:
        return None, None

    uri = str(schema_uri).strip()
    if not uri:
        return None, None

    run_id = fallback_run_id
    artifact_path: Optional[str] = None

    if uri.startswith("mlflow://") and mlflow_service:
        try:
            run_id_candidate, artifact_candidate = mlflow_service.parse_artifact_uri(uri)
        except ValueError:
            run_id_candidate, artifact_candidate = None, None
        if run_id_candidate:
            run_id = run_id_candidate
        if artifact_candidate:
            artifact_path = artifact_candidate

    if artifact_path is None and "artifacts/" in uri:
        artifact_path = uri.split("artifacts/", 1)[1].lstrip("/")

    if artifact_path is None and uri.startswith("runs:/"):
        remainder = uri[len("runs:/"):]
        if "/" in remainder:
            run_segment, artifact_segment = remainder.split("/", 1)
            run_id = run_segment or run_id
            artifact_path = artifact_segment
        else:
            run_id = remainder or run_id
            artifact_path = ""

    if artifact_path is None and not uri.startswith(("mlflow://", "runs:/", "s3://", "http://", "https://")):
        artifact_path = uri.lstrip("/")

    return run_id, artifact_path


def _download_schema_field_names(
    mlflow_service: Any,
    *,
    schema_uri: Optional[Any],
    run_id: Optional[str],
) -> Optional[List[str]]:
    if not mlflow_service:
        return None

    resolved_run_id, artifact_path = _resolve_schema_download_params(
        schema_uri,
        fallback_run_id=run_id,
        mlflow_service=mlflow_service,
    )

    if not resolved_run_id or artifact_path is None:
        return None

    artifact_path = artifact_path.strip("/")

    try:
        with tempfile.TemporaryDirectory(prefix="schema_") as tmp_dir:
            downloaded = Path(
                mlflow_service.client.download_artifacts(
                    resolved_run_id,
                    artifact_path,
                    dst_path=tmp_dir,
                )
            )

            target = downloaded
            if downloaded.is_dir():
                json_files = sorted(downloaded.glob("*.json"))
                if json_files:
                    target = json_files[0]
            if not target.exists() or target.is_dir():
                return None

            with target.open("r", encoding="utf-8") as handle:
                schema_payload = json.load(handle)
    except Exception as exc:  # pragma: no cover - network/io failures
        logger.warning(
            "[schema-fetch] Failed to download schema | run_id=%s path=%s error=%s",
            resolved_run_id,
            artifact_path,
            exc,
        )
        return None

    extracted = _collect_schema_column_names(schema_payload)
    return _deduplicate_names(extracted)


def _parse_signature_section(signature_payload: Any) -> tuple[Optional[List[str]], Optional[List[str]]]:
    if isinstance(signature_payload, str):
        candidate = signature_payload.strip()
        if not candidate:
            return None, None
        try:
            signature_payload = json.loads(candidate)
        except json.JSONDecodeError:
            return None, None

    if isinstance(signature_payload, dict):
        inputs = _deduplicate_names(_collect_schema_column_names(signature_payload.get("inputs")))
        outputs = _deduplicate_names(_collect_schema_column_names(signature_payload.get("outputs")))
        return inputs, outputs

    if isinstance(signature_payload, list):
        inputs = _deduplicate_names(_collect_schema_column_names(signature_payload))
        return inputs, None

    return None, None


def _extract_schema_from_mlmodel(
    mlflow_service: Any,
    *,
    run_id: Optional[str],
    artifact_path: Optional[str],
) -> tuple[Optional[List[str]], Optional[List[str]]]:
    if not mlflow_service or not run_id:
        return None, None

    normalized_path = (artifact_path or "").strip("/")

    try:
        with tempfile.TemporaryDirectory(prefix="mlmodel_schema_") as tmp_dir:
            downloaded = Path(
                mlflow_service.client.download_artifacts(
                    run_id,
                    normalized_path or "",
                    dst_path=tmp_dir,
                )
            )

            candidates: List[Path] = []
            if downloaded.is_file():
                if downloaded.name == "MLmodel":
                    candidates.append(downloaded)
                else:
                    mlmodel_candidate = downloaded.parent / "MLmodel"
                    if mlmodel_candidate.exists():
                        candidates.append(mlmodel_candidate)
            else:
                direct = downloaded / "MLmodel"
                if direct.exists():
                    candidates.append(direct)
                else:
                    candidates.extend(downloaded.rglob("MLmodel"))

            for mlmodel_path in candidates:
                try:
                    raw_text = mlmodel_path.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    continue

                payload: Any = None
                try:
                    import yaml  # type: ignore

                    payload = yaml.safe_load(raw_text)
                except Exception:
                    try:
                        payload = json.loads(raw_text)
                    except json.JSONDecodeError:
                        payload = None

                if not isinstance(payload, dict):
                    continue

                for key in ("signature", "model_signature", "signature_dict"):
                    inputs, outputs = _parse_signature_section(payload.get(key))
                    if inputs or outputs:
                        return inputs, outputs

                input_payload = payload.get("input_schema")
                output_payload = payload.get("output_schema")
                inferred_inputs = _deduplicate_names(_collect_schema_column_names(input_payload)) if input_payload else None
                inferred_outputs = _deduplicate_names(_collect_schema_column_names(output_payload)) if output_payload else None
                if inferred_inputs or inferred_outputs:
                    return inferred_inputs, inferred_outputs
    except Exception as exc:  # pragma: no cover - artifact download failures
        logger.debug(
            "[schema-fetch] Failed to inspect MLmodel for schema | run_id=%s path=%s error=%s",
            run_id,
            artifact_path,
            exc,
        )

    return None, None


def _resolve_schema_field_names(
    mlflow_service: Any,
    *,
    run_id: Optional[str],
    tags: Optional[Dict[str, Any]],
    artifact_path: Optional[str],
) -> tuple[Optional[List[str]], Optional[List[str]]]:
    tag_dict = tags if isinstance(tags, dict) else {}

    input_schema_fields, output_schema_fields = _extract_schema_from_mlmodel(
        mlflow_service,
        run_id=run_id,
        artifact_path=artifact_path,
    )

    if not input_schema_fields:
        input_schema_fields = _parse_literal_list(tag_dict.get("feature_names"))
    if not input_schema_fields:
        input_schema_fields = _parse_literal_list(tag_dict.get("input_feature_names"))
    if not input_schema_fields:
        input_schema_fields = _download_schema_field_names(
            mlflow_service,
            schema_uri=tag_dict.get("input_schema_uri"),
            run_id=run_id,
        )

    if not output_schema_fields:
        output_schema_fields = _parse_literal_list(tag_dict.get("task_original_classes"))
    if not output_schema_fields:
        output_schema_fields = _parse_literal_list(tag_dict.get("output_classes"))
    if not output_schema_fields:
        output_schema_fields = _download_schema_field_names(
            mlflow_service,
            schema_uri=tag_dict.get("output_schema_uri"),
            run_id=run_id,
        )

    return (
        _deduplicate_names(input_schema_fields or []),
        _deduplicate_names(output_schema_fields or []),
    )

def _build_mlflow_metadata_payload(model: Any) -> Dict[str, Any]:
    metadata = getattr(model, "metadata", None)
    if isinstance(metadata, str) and metadata:
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            metadata = {}
    if not isinstance(metadata, dict):
        return {}

    mlflow_block = metadata.get("mlflow")
    if not isinstance(mlflow_block, dict):
        return {}

    payload: Dict[str, Any] = {
        "tracking_uri": mlflow_block.get("tracking_uri"),
        "model_uri": mlflow_block.get("model_uri"),
        "run_id": mlflow_block.get("run_id"),
        "model_version": mlflow_block.get("model_version"),
        "registered_model_name": mlflow_block.get("registered_model_name"),
        "load_flavor": mlflow_block.get("load_flavor"),
        "artifact_path": mlflow_block.get("artifact_path"),
    }

    additional = mlflow_block.get("additional_metadata")
    if isinstance(additional, dict):
        processed_additional: Dict[str, Any] = {}
        for key, value in additional.items():
            parsed_value = _parse_possible_json(value)
            if key == "tags" and isinstance(parsed_value, dict):
                parsed_value = {
                    tag_key: _parse_possible_json(tag_value)
                    for tag_key, tag_value in parsed_value.items()
                }
            iso_key = None
            if key.endswith("_timestamp"):
                iso_value = _format_timestamp_ms(parsed_value if not isinstance(parsed_value, dict) else None)
                if iso_value:
                    iso_key = f"{key}_iso"
                    processed_additional[iso_key] = iso_value
            processed_additional[key] = parsed_value
        payload["additional_metadata"] = processed_additional
    elif additional is not None:
        payload["additional_metadata"] = _parse_possible_json(additional)

    return payload


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

    mlflow_service = _get_mlflow_service_or_none(request)
    if not mlflow_service:
        logger.error(
            "[upload_model] MLflow service unavailable | user_id=%s",
            user_id,
        )
        backend_log.error("MLflow service unavailable for upload")
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, detail="MLFLOW_SERVICE_UNAVAILABLE")

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

    safe_model_name = _sanitize_name(model_name)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    artifact_name_parts = [safe_model_name]
    if model_version:
        artifact_name_parts.append(_sanitize_name(model_version))
    artifact_name_parts.append(timestamp)
    artifact_filename = "__".join(artifact_name_parts) + suffix

    temp_path: Optional[Path] = None
    hasher = hashlib.sha256()
    total_bytes = 0

    try:
        with tempfile.NamedTemporaryFile(prefix="upload_", suffix=suffix, delete=False) as temp_file:
            temp_path = Path(temp_file.name)
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total_bytes += len(chunk)
                hasher.update(chunk)
                temp_file.write(chunk)
    except Exception as exc:  # pragma: no cover - IO failure path
        logger.exception(
            "[upload_model] Failed to buffer upload | user_id=%s",
            user_id,
        )
        if temp_path:
            temp_path.unlink(missing_ok=True)
        backend_log.error("Model upload failed during buffering", exception=exc)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="MODEL_UPLOAD_FAILED") from exc
    finally:
        await file.close()

    if total_bytes == 0 or not temp_path:
        logger.warning(
            "[upload_model] Empty upload detected | user_id=%s model_name=%s",
            user_id,
            model_name,
        )
        backend_log.warn("Empty model upload rejected", metadata={"model_name": model_name})
        if temp_path:
            temp_path.unlink(missing_ok=True)
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="EMPTY_MODEL_FILE")

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
        if temp_path:
            temp_path.unlink(missing_ok=True)
        backend_log.warn(
            "Duplicate artifact detected via checksum",
            metadata={
                "model_name": model_name,
                "checksum": checksum,
                "existing_model_id": duplicate_artifact.id,
            }
        )
        raise HTTPException(status.HTTP_409_CONFLICT, detail="MODEL_ARTIFACT_ALREADY_EXISTS")

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
        if temp_path:
            temp_path.unlink(missing_ok=True)
        backend_log.error(
            "Invalid MLflow run information",
            metadata={"error": str(run_error)},
        )
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="INVALID_MLFLOW_RUN") from run_error

    resolved_experiment_id: Optional[str] = mlflow_experiment_id or mlflow_service.default_experiment_id

    try:
        artifact_relative_path = mlflow_service.log_artifact(
            resolved_run_id,
            temp_path,
            artifact_dir=artifact_directory,
        )
    except Exception as mlflow_exc:
        logger.exception(
            "[upload_model] Failed to upload artifact to MLflow | user_id=%s run_id=%s",
            user_id,
            resolved_run_id,
        )
        if temp_path:
            temp_path.unlink(missing_ok=True)
        backend_log.error("Failed to upload artifact to MLflow", exception=mlflow_exc)
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, detail="MLFLOW_ARTIFACT_UPLOAD_FAILED") from mlflow_exc
    finally:
        if temp_path:
            temp_path.unlink(missing_ok=True)

    artifact_uri = mlflow_service.build_artifact_uri(resolved_run_id, artifact_relative_path)

    mlflow_metadata = {
        "mlflow_run_id": resolved_run_id,
        "mlflow_artifact_path": artifact_relative_path,
        "mlflow_artifact_uri": artifact_uri,
        "mlflow_tracking_uri": mlflow_service.tracking_uri,
        "mlflow_model_uri": artifact_uri,
    }
    if resolved_experiment_id:
        mlflow_metadata["mlflow_experiment_id"] = resolved_experiment_id
    if run_name_for_mlflow:
        mlflow_metadata["mlflow_run_name"] = run_name_for_mlflow
    if mlflow_service.default_experiment_id:
        mlflow_metadata.setdefault("mlflow_default_experiment_id", mlflow_service.default_experiment_id)

    parsed_metadata.update(mlflow_metadata)
    parsed_metadata.setdefault("artifact_sha256", checksum)
    parsed_metadata.setdefault("artifact_size_bytes", total_bytes)

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
        "mlflow_metadata": _build_mlflow_metadata_payload(created_model),
    }


@router.get("/")
@router.get("")
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
            "mlflow_metadata": _build_mlflow_metadata_payload(model),
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
        "mlflow_metadata": _build_mlflow_metadata_payload(model),
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

    logger.info(
        "[run_inference] Model resolved | model_id=%s name=%s version=%s framework=%s",
        model_record.id,
        model_record.model_name,
        model_record.model_version,
        model_record.framework,
    )

    mlflow_info = extract_mlflow_info(model_record, model_record.file_path)
    if mlflow_info:
        logger.info(
            "[run_inference] Extracted MLflow info | model_id=%s model_uri=%s run_id=%s artifact_path=%s",
            model_record.id,
            mlflow_info.model_uri,
            mlflow_info.run_id,
            mlflow_info.artifact_path,
        )
    else:
        logger.warning(
            "[run_inference] MLflow info missing from metadata | model_id=%s",
            model_record.id,
        )
    local_model_path: Optional[str] = None

    if mlflow_info and isinstance(mlflow_info.model_uri, str) and mlflow_info.model_uri.startswith(("models:/", "models://")):
        mlflow_service = _get_mlflow_service_or_none(request)
        if not mlflow_service:
            logger.error(
                "[run_inference] MLflow service unavailable for registry download | user_id=%s model_id=%s",
                user_id,
                model_record.id,
            )
            backend_log.error(
                "MLflow service unavailable for registry inference",
                metadata={"model_id": model_record.id},
            )
            raise HTTPException(status.HTTP_502_BAD_GATEWAY, detail="MLFLOW_SERVICE_UNAVAILABLE")

        raw_metadata = model_record.metadata
        metadata_dict = raw_metadata if isinstance(raw_metadata, dict) else {}
        registry_model_name = (
            mlflow_info.registered_model_name
            or metadata_dict.get("mlflow_registered_model_name")
            or model_record.model_name
        )
        registry_version = mlflow_info.model_version or model_record.model_version

        if not (registry_model_name and registry_version):
            logger.error(
                "[run_inference] Registry metadata incomplete | user_id=%s model_id=%s",
                user_id,
                model_record.id,
            )
            backend_log.error(
                "Missing registry metadata for inference",
                metadata={"model_id": model_record.id},
            )
            raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="MODEL_MLFLOW_METADATA_MISSING")

        storage_location = None
        if mlflow_info and isinstance(mlflow_info.extra, dict):
            storage_location = (
                mlflow_info.extra.get("storage_location")
                or mlflow_info.extra.get("download_uri")
                or mlflow_info.extra.get("artifact_uri")
            )
        if not storage_location and mlflow_info and mlflow_info.run_id:
            storage_location = mlflow_service.get_storage_location_for_run(mlflow_info.run_id)

        try:
            downloaded_path = mlflow_service.download_model_version_artifact(
                registry_model_name,
                registry_version,
                artifact_path=None,
                storage_location=storage_location,
                run_id=mlflow_info.run_id if mlflow_info else None,
            )
            resolved_file = mlflow_service.resolve_model_artifact_path(downloaded_path)
            if resolved_file.is_dir() and mlflow_info.artifact_path:
                downloaded_path = mlflow_service.download_model_version_artifact(
                    registry_model_name,
                    registry_version,
                    artifact_path=mlflow_info.artifact_path,
                    storage_location=storage_location,
                    run_id=mlflow_info.run_id if mlflow_info else None,
                )
                resolved_file = mlflow_service.resolve_model_artifact_path(downloaded_path)
            local_model_path = str(resolved_file)
            logger.info(
                "[run_inference] Downloaded registry model | user_id=%s model_id=%s path=%s",
                user_id,
                model_record.id,
                local_model_path,
            )
        except Exception as exc:
            logger.exception(
                "[run_inference] Failed to download registry model | user_id=%s model_id=%s",
                user_id,
                model_record.id,
            )
            backend_log.error(
                "Failed to download registry model",
                metadata={
                    "model_id": model_record.id,
                    "model_name": registry_model_name,
                    "model_version": registry_version,
                },
                exception=exc,
            )
            raise HTTPException(status.HTTP_502_BAD_GATEWAY, detail="MLFLOW_ARTIFACT_UNAVAILABLE") from exc
    else:
        run_id = mlflow_info.run_id if mlflow_info and mlflow_info.run_id else None
        artifact_path = mlflow_info.artifact_path if mlflow_info and mlflow_info.artifact_path else None

        if not (run_id and artifact_path):
            logger.info(
                "[run_inference] Falling back to legacy MLflow binding | model_id=%s",
                model_record.id,
            )
            resolved_run_id, resolved_artifact_path = _resolve_mlflow_binding(model_record.metadata, model_record.file_path)
            run_id = run_id or resolved_run_id
            artifact_path = artifact_path or resolved_artifact_path

        if not (run_id and artifact_path):
            logger.error(
                "[run_inference] Missing MLflow binding | user_id=%s model_id=%s",
                user_id,
                model_record.id,
            )
            backend_log.error(
                "Model metadata missing MLflow binding",
                metadata={"model_id": model_record.id},
            )
            raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="MODEL_MLFLOW_METADATA_MISSING")

        mlflow_service = _get_mlflow_service_or_none(request)
        if not mlflow_service:
            logger.error(
                "[run_inference] MLflow service unavailable | user_id=%s run_id=%s",
                user_id,
                run_id,
            )
            backend_log.error("MLflow service unavailable for inference", metadata={"run_id": run_id})
            raise HTTPException(status.HTTP_502_BAD_GATEWAY, detail="MLFLOW_SERVICE_UNAVAILABLE")

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

    if not local_model_path:
        logger.error(
            "[run_inference] Unable to resolve model path | user_id=%s model_id=%s",
            user_id,
            model_record.id,
        )
        backend_log.error(
            "Failed to resolve model path for inference",
            metadata={"model_id": model_record.id},
        )
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="MODEL_ARTIFACT_MISSING")

    try:
        inference_result = _inference_service.predict(
            file_path=local_model_path,
            inputs=payload.inputs,
            input_schema=model_record.get_input_schema(),
            return_probabilities=payload.return_probabilities,
        )
        logger.info(
            "[run_inference] Prediction finished | model_id=%s records=%s",
            model_record.id,
            len(payload.inputs),
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
            "mlflow_metadata": _build_mlflow_metadata_payload(model_record),
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
    removed_ids: List[int] = []
    imported_ids: List[int] = []
    updated_ids: List[int] = []

    mlflow_service = _get_mlflow_service_or_none(request)
    if not mlflow_service:
        logger.error(
            "[sync_model_artifacts] MLflow service unavailable | user_id=%s",
            user_id,
        )
        backend_log.error("MLflow service unavailable for sync")
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, detail="MLFLOW_SERVICE_UNAVAILABLE")

    existing_lookup: Dict[tuple[str, Optional[str]], Any] = {
        (model.model_name, model.model_version): model for model in models
    }

    try:
        registered_models = mlflow_service.list_registered_models()
    except Exception as exc:  # pragma: no cover - depends on MLflow backend
        logger.exception(
            "[sync_model_artifacts] Failed to query MLflow registry | user_id=%s",
            user_id,
        )
        backend_log.error("Failed to query MLflow registry", exception=exc)
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, detail="MLFLOW_REGISTRY_UNAVAILABLE") from exc

    remote_lookup: Dict[tuple[str, Optional[str]], Dict[str, Any]] = {}

    for registry_entry in registered_models:
        model_name = getattr(registry_entry, "name", None)
        if not model_name:
            continue
        try:
            versions = mlflow_service.list_model_versions(model_name)
        except Exception as exc:  # pragma: no cover - depends on MLflow backend
            logger.warning(
                "[sync_model_artifacts] Failed to list versions | model_name=%s error=%s",
                model_name,
                exc,
            )
            continue

        for version in versions:
            version_str = str(getattr(version, "version", "")) or None
            key = (model_name, version_str)

            run_id_candidate = getattr(version, "run_id", None)
            source = getattr(version, "source", None)
            parsed_run_id, artifact_path = _parse_registered_model_source(source, run_id=run_id_candidate)

            storage_location = getattr(version, "storage_location", None)
            if storage_location:
                storage_run_id, storage_artifact_path = _parse_registered_model_source(
                    storage_location,
                    run_id=parsed_run_id,
                )
                parsed_run_id = parsed_run_id or storage_run_id
                artifact_path = artifact_path or storage_artifact_path

            download_uri = None
            if version_str:
                download_uri = mlflow_service.get_model_version_download_uri(model_name, version_str)
                if download_uri:
                    dl_run_id, dl_artifact_path = _parse_registered_model_source(download_uri, run_id=parsed_run_id)
                    parsed_run_id = parsed_run_id or dl_run_id
                    artifact_path = artifact_path or dl_artifact_path

            if not parsed_run_id or artifact_path is None:
                logger.warning(
                    "[sync_model_artifacts] Skipping version without resolvable artifact | model=%s version=%s",
                    model_name,
                    version_str,
                )
                continue

            artifact_uri = mlflow_service.build_artifact_uri(parsed_run_id, artifact_path)
            model_uri = f"models:/{model_name}/{version_str}" if version_str else artifact_uri

            tags_dict = dict(getattr(version, "tags", {}) or {})
            input_schema_fields, output_schema_fields = _resolve_schema_field_names(
                mlflow_service,
                run_id=parsed_run_id,
                tags=tags_dict,
                artifact_path=artifact_path,
            )

            remote_lookup[key] = {
                "model_name": model_name,
                "model_version": version_str,
                "run_id": parsed_run_id,
                "artifact_path": artifact_path,
                "artifact_uri": artifact_uri,
                "model_uri": model_uri,
                "status": getattr(version, "status", None),
                "stage": getattr(version, "current_stage", None),
                "source": source,
                "run_link": getattr(version, "run_link", None),
                "description": getattr(version, "description", None) or getattr(registry_entry, "description", None),
                "tags": tags_dict,
                "creation_timestamp": getattr(version, "creation_timestamp", None),
                "last_updated_timestamp": getattr(version, "last_updated_timestamp", None),
                "download_uri": download_uri,
                "storage_location": storage_location,
                "input_schema_fields": input_schema_fields,
                "output_schema_fields": output_schema_fields,
            }

    # Remove models that no longer exist in MLflow registry
    for key, model in existing_lookup.items():
        remote_info = remote_lookup.get(key)
        if not remote_info:
            logger.warning(
                "[sync_model_artifacts] Removing stale registry entry | model_id=%s name=%s version=%s",
                model.id,
                model.model_name,
                model.model_version,
            )
            if deletion_service.delete(model.id, model=model, remove_artifact=False):
                removed_ids.append(model.id)
            continue

        if not mlflow_service.artifact_exists(remote_info["run_id"], remote_info["artifact_path"]):
            logger.warning(
                "[sync_model_artifacts] Missing MLflow artifact detected | model_id=%s run_id=%s path=%s",
                model.id,
                remote_info["run_id"],
                remote_info["artifact_path"],
            )
            if deletion_service.delete(model.id, model=model, remove_artifact=False):
                removed_ids.append(model.id)
            continue

        staged_changes = False
        if model.file_path != remote_info["model_uri"]:
            model.file_path = remote_info["model_uri"]
            staged_changes = True

        remote_input_schema = remote_info.get("input_schema_fields")
        if remote_input_schema is not None and model.input_schema != remote_input_schema:
            model.input_schema = remote_input_schema
            staged_changes = True

        remote_output_schema = remote_info.get("output_schema_fields")
        if remote_output_schema is not None and model.output_schema != remote_output_schema:
            model.output_schema = remote_output_schema
            staged_changes = True

        metadata = model.metadata if isinstance(model.metadata, dict) else {}
        mlflow_meta = metadata.get("mlflow") if isinstance(metadata, dict) else None
        if not isinstance(mlflow_meta, dict):
            mlflow_meta = {}

        desired_mlflow_meta = {
            "tracking_uri": mlflow_service.tracking_uri,
            "model_uri": remote_info["model_uri"],
            "run_id": remote_info["run_id"],
            "model_version": remote_info["model_version"],
            "registered_model_name": remote_info["model_name"],
            "load_flavor": "pyfunc",
            "artifact_path": remote_info["artifact_path"],
            "additional_metadata": {
                "artifact_uri": remote_info["artifact_uri"],
                "status": remote_info["status"],
                "stage": remote_info["stage"],
                "source": remote_info["source"],
                "run_link": remote_info["run_link"],
                "tags": remote_info["tags"],
                "creation_timestamp": remote_info["creation_timestamp"],
                "last_updated_timestamp": remote_info["last_updated_timestamp"],
                "download_uri": remote_info.get("download_uri"),
                "storage_location": remote_info.get("storage_location"),
            },
        }

        desired_mlflow_meta["additional_metadata"] = {
            key_name: value
            for key_name, value in desired_mlflow_meta["additional_metadata"].items()
            if value not in (None, {}, [])
        }

        if mlflow_meta != desired_mlflow_meta:
            metadata["mlflow"] = desired_mlflow_meta
            model.metadata = metadata
            staged_changes = True

        if staged_changes and registry_service.update_model(model):
            updated_ids.append(model.id)

    # Import new MLflow registry entries
    for key, remote_info in remote_lookup.items():
        if key in existing_lookup:
            continue

        metadata_payload: Dict[str, Any] = {
            "mlflow": {
                "additional_metadata": {
                    "artifact_uri": remote_info["artifact_uri"],
                "status": remote_info["status"],
                "stage": remote_info["stage"],
                "source": remote_info["source"],
                "run_link": remote_info["run_link"],
                "tags": remote_info["tags"],
                "creation_timestamp": remote_info["creation_timestamp"],
                "last_updated_timestamp": remote_info["last_updated_timestamp"],
                "download_uri": remote_info.get("download_uri"),
                "storage_location": remote_info.get("storage_location"),
            }
        }
        }

        additional_meta = metadata_payload["mlflow"]["additional_metadata"]
        metadata_payload["mlflow"]["additional_metadata"] = {
            key_name: value for key_name, value in additional_meta.items() if value not in (None, {}, [])
        }

        try:
            created_model = registry_service.register_mlflow_model(
                model_name=remote_info["model_name"],
                model_uri=remote_info["model_uri"],
                tracking_uri=mlflow_service.tracking_uri,
                model_version=remote_info["model_version"],
                description=remote_info["description"],
                run_id=remote_info["run_id"],
                registered_model_name=remote_info["model_name"],
                load_flavor="pyfunc",
                artifact_path=remote_info["artifact_path"],
                input_schema=remote_info.get("input_schema_fields"),
                output_schema=remote_info.get("output_schema_fields"),
                metadata=metadata_payload,
                status="active",
            )
        except Exception as exc:  # pragma: no cover - persistence failure
            logger.exception(
                "[sync_model_artifacts] Failed to import MLflow model | model=%s version=%s",
                remote_info["model_name"],
                remote_info["model_version"],
            )
            backend_log.error(
                "Failed to import MLflow model",
                metadata={
                    "model_name": remote_info["model_name"],
                    "model_version": remote_info["model_version"],
                },
                exception=exc,
            )
            continue

        imported_ids.append(created_model.id)
        existing_lookup[key] = created_model

    logger.info(
        "[sync_model_artifacts] Sync complete | user_id=%s checked=%s removed=%s imported=%s updated=%s",
        user_id,
        len(models),
        len(removed_ids),
        len(imported_ids),
        len(updated_ids),
    )
    backend_log.success(
        "Model artifact sync completed",
        metadata={
            "checked": len(models),
            "removed": len(removed_ids),
            "imported": len(imported_ids),
            "updated": len(updated_ids),
        }
    )

    return {
        "checked": len(models),
        "removed": len(removed_ids),
        "removed_ids": removed_ids,
        "imported": len(imported_ids),
        "imported_ids": imported_ids,
        "updated": len(updated_ids),
        "updated_ids": updated_ids,
    }


@router.post("/{model_id}/stage")
async def update_model_stage(
    request: Request,
    model_id: int,
    payload: ModelStageUpdateRequest,
):
    user_id = extract_user_id_from_request(request)
    app_db = get_db_manager(request)
    backend_log = create_logger(app_db, _coerce_user_id(user_id), request)

    logger.info(
        "[update_model_stage] Stage change requested | user_id=%s model_id=%s target_stage=%s",
        user_id,
        model_id,
        payload.stage,
    )

    registry_service = ModelRegistryService(app_db)
    model = registry_service.get_model_by_id(model_id)
    if not model:
        logger.warning(
            "[update_model_stage] Model not found | user_id=%s model_id=%s",
            user_id,
            model_id,
        )
        backend_log.warn(
            "Stage update requested for missing model",
            metadata={"model_id": model_id},
        )
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="MODEL_NOT_FOUND")

    mlflow_service = _get_mlflow_service_or_none(request)
    if not mlflow_service:
        logger.error(
            "[update_model_stage] MLflow service unavailable | user_id=%s model_id=%s",
            user_id,
            model_id,
        )
        backend_log.error("MLflow service unavailable for stage update")
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, detail="MLFLOW_SERVICE_UNAVAILABLE")

    mlflow_info = extract_mlflow_info(model, model.file_path)
    metadata_payload = _build_mlflow_metadata_payload(model)

    registered_model_name = None
    if mlflow_info and mlflow_info.registered_model_name:
        registered_model_name = mlflow_info.registered_model_name
    elif metadata_payload.get("registered_model_name"):
        registered_model_name = metadata_payload["registered_model_name"]
    additional_meta = metadata_payload.get("additional_metadata") if metadata_payload else None

    model_version = None
    if mlflow_info and mlflow_info.model_version:
        model_version = mlflow_info.model_version
    elif metadata_payload.get("model_version"):
        model_version = metadata_payload["model_version"]
    elif model.model_version:
        model_version = model.model_version

    if not registered_model_name or not model_version:
        logger.error(
            "[update_model_stage] Missing registry identifiers | user_id=%s model_id=%s",
            user_id,
            model_id,
        )
        backend_log.error(
            "Stage update unsupported; missing registry identifiers",
            metadata={"model_id": model_id},
        )
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="MODEL_STAGE_UPDATE_UNAVAILABLE")

    previous_stage = None
    if isinstance(additional_meta, dict):
        previous_stage = additional_meta.get("stage")

    try:
        mlflow_service.transition_model_stage(
            model_name=registered_model_name,
            version=model_version,
            stage=payload.stage,
            archive_existing_versions=payload.archive_existing_versions,
        )
    except Exception as exc:  # pragma: no cover - depends on MLflow backend
        logger.exception(
            "[update_model_stage] Failed to transition stage | user_id=%s model_id=%s",
            user_id,
            model_id,
        )
        backend_log.error(
            "Failed to transition MLflow stage",
            metadata={
                "model_id": model_id,
                "model_name": registered_model_name,
                "model_version": model_version,
                "stage": payload.stage,
            },
            exception=exc,
        )
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, detail="MLFLOW_STAGE_TRANSITION_FAILED") from exc

    metadata_dict: Dict[str, Any]
    metadata_dict = model.metadata if isinstance(model.metadata, dict) else {}
    mlflow_meta = metadata_dict.get("mlflow")
    if not isinstance(mlflow_meta, dict):
        mlflow_meta = {}
    additional = mlflow_meta.get("additional_metadata")
    if not isinstance(additional, dict):
        additional = {}
    additional["stage"] = payload.stage
    mlflow_meta["additional_metadata"] = {
        key: value for key, value in additional.items() if value not in (None, {}, [])
    }
    mlflow_meta.setdefault("registered_model_name", registered_model_name)
    mlflow_meta.setdefault("model_version", model_version)
    metadata_dict["mlflow"] = mlflow_meta
    model.metadata = metadata_dict

    if not registry_service.update_model(model):
        logger.error(
            "[update_model_stage] Failed to persist stage | user_id=%s model_id=%s",
            user_id,
            model_id,
        )
        backend_log.error(
            "Failed to persist stage change",
            metadata={"model_id": model_id},
        )
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="MODEL_STAGE_UPDATE_FAILED")

    updated_metadata = _build_mlflow_metadata_payload(model)

    logger.info(
        "[update_model_stage] Stage updated | user_id=%s model_id=%s stage=%s",
        user_id,
        model_id,
        payload.stage,
    )
    backend_log.success(
        "Model stage updated",
        metadata={
            "model_id": model_id,
            "target_stage": payload.stage,
            "previous_stage": previous_stage,
            "archive_existing_versions": payload.archive_existing_versions,
        },
    )

    return {
        "model_id": model.id,
        "model_name": model.model_name,
        "model_version": model.model_version,
        "registered_model_name": registered_model_name,
        "stage": payload.stage,
        "previous_stage": previous_stage,
        "archive_existing_versions": payload.archive_existing_versions,
        "mlflow_metadata": updated_metadata,
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

    metadata_payload = _build_mlflow_metadata_payload(existing)
    additional_meta = metadata_payload.get("additional_metadata") if isinstance(metadata_payload, dict) else None
    current_stage = None
    if isinstance(additional_meta, dict):
        stage_value = additional_meta.get("stage")
        if isinstance(stage_value, str):
            current_stage = stage_value.strip()

    if current_stage and current_stage.lower() == "production":
        logger.warning(
            "[delete_model] Deletion blocked due to Production stage | user_id=%s model_id=%s",
            user_id,
            model_id,
        )
        backend_log.warn(
            "Attempted deletion of Production stage model",
            metadata={"model_id": model_id, "stage": current_stage},
        )
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="MODEL_STAGE_DELETE_BLOCKED")

    mlflow_service = None
    try:
        mlflow_service = get_mlflow_service(request)
    except HTTPException:
        mlflow_service = None

    mlflow_info = extract_mlflow_info(existing, existing.file_path)

    run_id = None
    artifact_path = None
    if mlflow_info and mlflow_info.run_id and mlflow_info.artifact_path is not None:
        run_id = mlflow_info.run_id
        artifact_path = mlflow_info.artifact_path
    else:
        fallback_run_id, fallback_artifact = _resolve_mlflow_binding(existing.metadata, existing.file_path)
        run_id = fallback_run_id
        artifact_path = fallback_artifact

    is_registry_model = False
    registry_model_name: Optional[str] = None
    registry_version: Optional[str] = None

    if mlflow_info and (mlflow_info.registered_model_name or (mlflow_info.model_uri and mlflow_info.model_uri.startswith("models:/"))):
        registry_model_name = mlflow_info.registered_model_name or existing.model_name
        registry_version = mlflow_info.model_version or existing.model_version
        is_registry_model = True
    elif isinstance(metadata_payload, dict):
        registry_model_name = metadata_payload.get("registered_model_name") or existing.model_name
        registry_version = metadata_payload.get("model_version") or existing.model_version
        if metadata_payload.get("model_uri", "").startswith("models:/"):
            is_registry_model = True

    if is_registry_model:
        if not registry_model_name or not registry_version:
            logger.error(
                "[delete_model] Missing registry identifiers for delete | user_id=%s model_id=%s",
                user_id,
                model_id,
            )
            backend_log.error(
                "Registry delete failed; missing identifiers",
                metadata={"model_id": model_id},
            )
            raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="MODEL_DELETE_METADATA_MISSING")

        if not mlflow_service:
            logger.error(
                "[delete_model] MLflow service unavailable for registry delete | user_id=%s model_id=%s",
                user_id,
                model_id,
            )
            backend_log.error(
                "MLflow service unavailable; cannot delete model version",
                metadata={"model_id": model_id, "registered_model_name": registry_model_name, "model_version": registry_version},
            )
            raise HTTPException(status.HTTP_502_BAD_GATEWAY, detail="MLFLOW_SERVICE_UNAVAILABLE")

        try:
            mlflow_service.delete_model_version(registry_model_name, registry_version)
            logger.info(
                "[delete_model] Deleted MLflow registry version | user_id=%s model_id=%s model=%s version=%s",
                user_id,
                model_id,
                registry_model_name,
                registry_version,
            )
        except Exception as exc:  # pragma: no cover - depends on MLflow backend
            logger.exception(
                "[delete_model] Failed to delete MLflow model version | user_id=%s model_id=%s model=%s version=%s",
                user_id,
                model_id,
                registry_model_name,
                registry_version,
            )
            backend_log.error(
                "Failed to delete MLflow model version",
                metadata={
                    "model_id": model_id,
                    "model_name": registry_model_name,
                    "model_version": registry_version,
                },
                exception=exc,
            )
            raise HTTPException(status.HTTP_502_BAD_GATEWAY, detail="MLFLOW_MODEL_VERSION_DELETE_FAILED") from exc

    if run_id and artifact_path and not mlflow_service and not is_registry_model:
        logger.error(
            "[delete_model] MLflow service unavailable; cannot delete artifact | user_id=%s model_id=%s",
            user_id,
            model_id,
        )
        backend_log.error(
            "MLflow service unavailable; remote artifact not deleted",
            metadata={"model_id": model_id, "run_id": run_id},
        )
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, detail="MLFLOW_SERVICE_UNAVAILABLE")

    if run_id and artifact_path and mlflow_service:
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
            logger.warning(
                "[delete_model] Failed to delete MLflow artifact (continuing) | user_id=%s model_id=%s run_id=%s path=%s error=%s",
                user_id,
                model_id,
                run_id,
                artifact_path,
                exc,
            )
            backend_log.warn(
                "Failed to delete MLflow artifact during model delete",
                metadata={
                    "model_id": model_id,
                    "run_id": run_id,
                    "artifact_path": artifact_path,
                },
            )

    try:
        deleted = deletion_service.delete(
            model_id,
            model=existing,
            remove_artifact=False,
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

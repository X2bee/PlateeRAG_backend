import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

from editor.node_composer import Node
from editor.utils.helper.service_helper import AppServiceManager
from langchain_core.tools import tool
from pydantic import BaseModel, Field, ValidationError, create_model
from fastapi import Request

from service.model.model_inference_service import ModelInferenceService
from service.model.model_registry_service import ModelRegistryService
from service.model.mlflow_utils import extract_mlflow_info
from controller.model.modelController import list_models
from editor.utils.helper.async_helper import sync_run_async

logger = logging.getLogger(__name__)
_inference_service = ModelInferenceService()


class MachineLearningTool(Node):
    categoryId = "xgen"
    functionId = "ml"
    nodeId = "ml/MachineLearningTool"
    nodeName = "Machine Learning Tool"
    description = "머신 러닝 모델을 호출하여 예측 결과를 반환하는 Tool 노드입니다."
    tags = ["ml", "tool", "prediction", "inference"]

    inputs: List[dict] = []
    outputs = [
        {"id": "tools", "name": "Tools", "type": "TOOL"},
    ]

    parameters = [
        {"id": "tool_name", "name": "Tool Name", "type": "STR", "value": "ml_prediction_tool", "required": True},
        {"id": "description", "name": "Description", "type": "STR", "value": "Use this tool to run predictions with the configured machine learning model.", "required": True, "description": "이 도구를 언제 호출해야 하는지 설명합니다."},
        {"id": "model_identifier", "name": "Model Identifier", "type": "STR", "value": "", "required": False, "optional": True, "description": "등록된 모델의 ID 또는 이름입니다. 숫자만 입력하면 ID로 간주합니다.", "is_api": True, "api_name": "api_model_identifier", "options": []},
        {"id": "model_version", "name": "Model Version", "type": "STR", "value": "", "required": False, "optional": True, "description": "모델 이름으로 조회할 때 사용할 버전 정보입니다."},
        {"id": "model_file_path", "name": "Model File Path", "type": "STR", "value": "", "required": False, "optional": True, "description": "직접 지정할 모델 아티팩트 경로입니다. 설정되어 있으면 DB 조회보다 우선합니다."},
        {"id": "default_return_probabilities", "name": "Default Return Probabilities", "type": "BOOL", "value": False, "required": False, "optional": True, "description": "모델이 지원하는 경우 확률을 기본으로 반환할지 여부입니다."},
        {"id": "default_return_dict", "name": "Default Return Dict", "type": "BOOL", "value": False, "required": False, "optional": True, "description": "기본 응답을 문자열이 아닌 딕셔너리로 반환할지 여부입니다."},
        {"id": "enable_batch_predictions", "name": "Enable Batch Predictions", "type": "BOOL", "value": True, "required": False, "optional": True, "description": "batch_records 입력을 통해 다중 예측을 지원할지 여부입니다."},
    ]

    @staticmethod
    def _build_record_model(
        model_record: Optional[Any],
    ) -> Tuple[Optional[Type[BaseModel]], List[str]]:
        if not model_record:
            return None, []

        input_schema: Optional[List[str]] = None
        getter = getattr(model_record, "get_input_schema", None)
        if callable(getter):
            try:
                input_schema = getter()
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.warning("Failed to read input schema from model record: %s", exc)

        if not input_schema:
            input_schema = getattr(model_record, "input_schema", None)

        if not input_schema:
            return None, []

        sanitized_fields: Dict[str, Tuple[Any, Any]] = {}
        ordered_fields: List[str] = []
        for index, field_name in enumerate(input_schema):
            if not isinstance(field_name, str):
                field_name = str(field_name)

            candidate = field_name.strip()
            if not candidate:
                candidate = f"feature_{index}"

            if not candidate.isidentifier():
                logger.warning(
                    "Input schema feature '%s' is not a valid identifier. Falling back to generic record input.",
                    field_name,
                )
                return None, []

            sanitized_fields[candidate] = (
                Any,
                Field(..., description=f"Value for feature '{field_name}'"),
            )
            ordered_fields.append(candidate)

        if not sanitized_fields:
            return None, []

        model_label = getattr(model_record, "model_name", "Model") or "Model"
        model_label = "".join(ch for ch in model_label if ch.isalnum()) or "Model"
        record_model = create_model(
            f"{model_label}InputRecord",
            **sanitized_fields,
        )

        return record_model, ordered_fields

    @staticmethod
    def _augment_args_schema(
        record_model: Optional[Type[BaseModel]],
        default_return_probabilities: bool,
        default_return_dict: bool,
        allow_batch: bool,
    ) -> Tuple[Type[BaseModel], List[str]]:
        if record_model is None:
            base_schema = create_model(
                "MachineLearningDefaultRecord",
                record=(Dict[str, Any], Field(..., description="예측에 사용할 피처 키-값 쌍입니다.")),
            )
            feature_fields = ["record"]
        else:
            base_schema = record_model
            feature_fields = list(base_schema.model_fields.keys())

        extra_fields: Dict[str, Tuple[Any, Any]] = {
            "return_probabilities": (bool, Field(default=default_return_probabilities, description="모델이 지원하는 경우 확률 값을 포함할지 여부.")),
            "return_dict": (bool, Field(default=default_return_dict, description="응답을 문자열이 아닌 딕셔너리로 반환할지 여부.")),
        }

        if allow_batch:
            extra_fields["batch_records"] = (
                Optional[List[Dict[str, Any]]],
                Field(default=None, description="예측할 레코드 리스트입니다. 지정하면 단일 레코드 입력 대신 사용합니다."),
            )

        augmented_schema = create_model(
            f"{base_schema.__name__}MachineLearningArgs",
            __base__=base_schema,
            **extra_fields,
        )

        return augmented_schema, feature_fields

    def api_model_identifier(self, request: Request) -> List[Dict[str, Any]]:
        try:
            response = sync_run_async(list_models(request))
        except Exception as exc:
            logger.error("Failed to fetch model list for selector: %s", exc)
            return []

        items: List[Dict[str, Any]]
        if isinstance(response, dict):
            items = response.get("items", []) or []
        elif isinstance(response, list):
            items = response
        else:
            items = []

        options: List[Dict[str, Any]] = []
        for model in items:
            if not isinstance(model, dict):
                continue
            model_id = model.get("id")
            model_name = model.get("model_name") or "Unnamed Model"
            model_version = model.get("model_version") or ""
            model_stage = model.get("stage") or ""
            if model_stage != 'Production':
                continue
            label = model_name
            if model_version:
                label = f"{model_name} (v{model_version})"

            if model_id is not None:
                value = str(model_id)
            else:
                value = model_name

            options.append({"value": value, "label": label})

        return options

    @staticmethod
    def _parse_registered_model_source(
        source: Optional[str],
        *,
        run_id: Optional[str] = None,
    ) -> Tuple[Optional[str], Optional[str]]:
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

    @staticmethod
    def _parse_mlflow_uri(uri: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
        if not isinstance(uri, str):
            return None, None
        prefix = "mlflow://"
        if not uri.startswith(prefix):
            return None, None
        remainder = uri[len(prefix):]
        if "/" in remainder:
            run_id, artifact_path = remainder.split("/", 1)
        else:
            run_id, artifact_path = remainder, ""
        return run_id or None, artifact_path.strip("/") if artifact_path else ""

    @classmethod
    def _resolve_mlflow_run_binding(
        cls,
        model_record: Optional[Any],
        mlflow_model_uri: Optional[str],
        run_id: Optional[str],
        artifact_path: Optional[str],
    ) -> Tuple[Optional[str], Optional[str]]:
        resolved_run_id = run_id
        resolved_artifact_path = artifact_path

        metadata = getattr(model_record, "metadata", None)
        artifact_uri = None
        storage_location = None

        if isinstance(metadata, dict):
            resolved_run_id = resolved_run_id or metadata.get("mlflow_run_id")
            resolved_artifact_path = resolved_artifact_path or metadata.get("mlflow_artifact_path")
            artifact_uri = metadata.get("mlflow_artifact_uri")
            storage_location = metadata.get("mlflow_storage_location")

            mlflow_block = metadata.get("mlflow")
            if isinstance(mlflow_block, dict):
                resolved_run_id = resolved_run_id or mlflow_block.get("run_id")
                resolved_artifact_path = resolved_artifact_path or mlflow_block.get("artifact_path")
                artifact_uri = artifact_uri or mlflow_block.get("artifact_uri")
                storage_location = storage_location or mlflow_block.get("storage_location")

                additional = mlflow_block.get("additional_metadata")
                if isinstance(additional, dict):
                    resolved_run_id = resolved_run_id or additional.get("run_id")
                    resolved_artifact_path = resolved_artifact_path or additional.get("artifact_path")
                    artifact_uri = artifact_uri or additional.get("artifact_uri")
                    storage_location = storage_location or additional.get("storage_location")

            if artifact_uri and not resolved_artifact_path:
                parsed_run, parsed_path = cls._parse_registered_model_source(artifact_uri, run_id=resolved_run_id)
                resolved_run_id = resolved_run_id or parsed_run
                resolved_artifact_path = resolved_artifact_path or parsed_path

            if storage_location and not resolved_artifact_path:
                parsed_run, parsed_path = cls._parse_registered_model_source(storage_location, run_id=resolved_run_id)
                resolved_run_id = resolved_run_id or parsed_run
                resolved_artifact_path = resolved_artifact_path or parsed_path

        if not resolved_artifact_path and mlflow_model_uri:
            parsed_run, parsed_path = cls._parse_registered_model_source(mlflow_model_uri, run_id=resolved_run_id)
            resolved_run_id = resolved_run_id or parsed_run
            resolved_artifact_path = resolved_artifact_path or parsed_path

        if not resolved_run_id or resolved_artifact_path is None:
            parsed_run, parsed_path = cls._parse_mlflow_uri(mlflow_model_uri)
            resolved_run_id = resolved_run_id or parsed_run
            if resolved_artifact_path is None:
                resolved_artifact_path = parsed_path

        return resolved_run_id, resolved_artifact_path

    def _ensure_local_model_path(
        self,
        model_record: Optional[Any],
        raw_path: Optional[str],
    ) -> Tuple[Optional[str], Optional[str]]:
        if not raw_path:
            return None, "Model artifact path is not configured."

        mlflow_info = extract_mlflow_info(model_record, raw_path)
        if not mlflow_info:
            return raw_path, None

        mlflow_service = AppServiceManager.get_mlflow_service()
        if mlflow_service is None:
            return None, "MLflow service is not available. Ensure the server exposes mlflow_service."

        model_uri = mlflow_info.model_uri or raw_path
        raw_metadata = getattr(model_record, "metadata", None)
        metadata_dict = raw_metadata if isinstance(raw_metadata, dict) else {}

        try:
            if isinstance(model_uri, str) and model_uri.startswith(("models:/", "models://")):
                registry_model_name = (
                    mlflow_info.registered_model_name
                    or (mlflow_info.extra.get("registered_model_name") if mlflow_info.extra else None)
                    or metadata_dict.get("mlflow_registered_model_name")
                    or getattr(model_record, "model_name", None)
                )
                registry_version = (
                    mlflow_info.model_version
                    or (mlflow_info.extra.get("model_version") if mlflow_info.extra else None)
                    or metadata_dict.get("mlflow_model_version")
                    or getattr(model_record, "model_version", None)
                )

                if not (registry_model_name and registry_version):
                    return None, "MLflow registry metadata is incomplete for this model."

                storage_location = None
                if mlflow_info.extra:
                    storage_location = (
                        mlflow_info.extra.get("storage_location")
                        or mlflow_info.extra.get("download_uri")
                        or mlflow_info.extra.get("artifact_uri")
                    )
                if not storage_location:
                    storage_location = (
                        metadata_dict.get("mlflow_storage_location")
                        or metadata_dict.get("mlflow_download_uri")
                        or metadata_dict.get("mlflow_artifact_uri")
                    )

                run_id = mlflow_info.run_id
                if not storage_location and run_id:
                    storage_location = mlflow_service.get_storage_location_for_run(run_id)

                downloaded_path = mlflow_service.download_model_version_artifact(
                    registry_model_name,
                    registry_version,
                    artifact_path=None,
                    storage_location=storage_location,
                    run_id=run_id,
                )
                resolved_file = mlflow_service.resolve_model_artifact_path(downloaded_path)
                if resolved_file.is_dir() and mlflow_info.artifact_path:
                    downloaded_path = mlflow_service.download_model_version_artifact(
                        registry_model_name,
                        registry_version,
                        artifact_path=mlflow_info.artifact_path,
                        storage_location=storage_location,
                        run_id=run_id,
                    )
                    resolved_file = mlflow_service.resolve_model_artifact_path(downloaded_path)
                return str(resolved_file), None

            resolved_run_id, resolved_artifact_path = self._resolve_mlflow_run_binding(
                model_record,
                model_uri,
                mlflow_info.run_id,
                mlflow_info.artifact_path,
            )

            if not resolved_run_id or resolved_artifact_path is None:
                return None, "Model metadata missing MLflow binding."

            artifact_uri = mlflow_service.build_artifact_uri(resolved_run_id, resolved_artifact_path or "")
            cached_path = mlflow_service.ensure_local_artifact(
                artifact_uri,
                expected_checksum=getattr(model_record, "file_checksum", None),
            )

            resolved_path = Path(str(cached_path))
            if resolved_path.is_dir():
                resolved_path = mlflow_service.resolve_model_artifact_path(resolved_path)

            return str(resolved_path), None

        except FileNotFoundError as exc:
            logger.exception("MLflow artifact not found: %s", exc)
            return None, f"MLflow artifact not found: {exc}"
        except Exception as exc:  # pragma: no cover - depends on MLflow backend
            logger.exception("Failed to prepare MLflow artifact: %s", exc)
            return None, f"Failed to prepare MLflow artifact: {exc}"

    @staticmethod
    def _resolve_model(
        registry_service: Optional[ModelRegistryService],
        model_identifier: str,
        model_version: str,
        model_file_path: str,
    ) -> Tuple[Optional[Any], Optional[str]]:
        file_path = (model_file_path or "").strip()
        model_record = None
        identifier = (model_identifier or "").strip()

        if registry_service and identifier:
            numeric_identifier: Optional[int] = None
            try:
                numeric_identifier = int(identifier)
            except (TypeError, ValueError):
                numeric_identifier = None

            if numeric_identifier is not None:
                try:
                    model_record = registry_service.get_model_by_id(numeric_identifier)
                except Exception as exc:  # pragma: no cover - defensive logging
                    logger.warning("Failed to fetch model by id %s: %s", numeric_identifier, exc)

            if model_record is None:
                try:
                    model_record = registry_service.get_model_by_name(identifier, model_version or None)
                except Exception as exc:  # pragma: no cover - defensive logging
                    logger.warning("Failed to fetch model by name %s: %s", identifier, exc)

        if model_record and not file_path:
            file_path = getattr(model_record, "file_path", "") or ""

        return model_record, file_path or None

    @staticmethod
    def _format_response(response: Dict[str, Any], return_dict: bool) -> Any:
        if return_dict:
            return response
        return json.dumps(response, ensure_ascii=False, indent=2)

    def _get_model_registry_service(self) -> Optional[ModelRegistryService]:
        try:
            db_manager = AppServiceManager.get_db_manager()
        except Exception as exc:  # pragma: no cover - service helpers handle availability
            logger.debug("Database manager is not available: %s", exc)
            return None

        if not db_manager:
            return None

        try:
            return ModelRegistryService(db_manager)
        except Exception as exc:  # pragma: no cover - database initialization issues
            logger.warning("Failed to create ModelRegistryService: %s", exc)
            return None

    def execute(
        self,
        tool_name: str,
        description: str,
        model_identifier: str = "",
        model_version: str = "",
        model_file_path: str = "",
        default_return_probabilities: bool = False,
        default_return_dict: bool = False,
        enable_batch_predictions: bool = True,
        *args,
        **kwargs,
    ):
        allow_batch = bool(enable_batch_predictions)

        guidance_lines = [
            "return_dict가 False이면 문자열(JSON) 형태로 응답을 받습니다.",
        ]
        if allow_batch:
            guidance_lines.append("batch_records를 전달하면 여러 레코드를 한 번에 예측합니다.")
        description = description + "\n" + "\n".join(guidance_lines)

        registry_service = self._get_model_registry_service()
        initial_record, initial_file_path = self._resolve_model(
            registry_service,
            model_identifier,
            model_version,
            model_file_path,
        )

        record_model, feature_fields = self._build_record_model(initial_record)
        actual_args_schema, feature_fields = self._augment_args_schema(
            record_model,
            default_return_probabilities,
            default_return_dict,
            allow_batch,
        )

        def resolve_model() -> Tuple[Optional[Any], Optional[str]]:
            local_registry = registry_service or self._get_model_registry_service()
            record, resolved_path = self._resolve_model(
                local_registry,
                model_identifier,
                model_version,
                model_file_path,
            )
            record = record or initial_record
            resolved_path = resolved_path or initial_file_path
            return record, resolved_path

        def get_output_schema(current_record: Optional[Any]) -> Optional[List[str]]:
            if current_record is None:
                return None
            schema: Optional[List[str]] = None
            getter = getattr(current_record, "get_output_schema", None)
            if callable(getter):
                try:
                    schema = getter()
                except Exception as exc:  # pragma: no cover - defensive guard
                    logger.debug("Failed to get output schema via accessor: %s", exc)
            if not schema:
                schema = getattr(current_record, "output_schema", None)
            if schema and isinstance(schema, list):
                return [str(item) for item in schema]
            return None

        def normalize_batch_records(
            raw_records: Optional[List[Dict[str, Any]]],
            validator_model: Optional[Type[BaseModel]] = None,
        ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
            normalized: List[Dict[str, Any]] = []
            if not raw_records:
                return normalized, None

            active_validator = validator_model or record_model

            for idx, record in enumerate(raw_records):
                if not isinstance(record, dict):
                    return [], f"Batch record at index {idx} must be a dictionary."
                if active_validator is not None:
                    try:
                        parsed = active_validator(**record)
                        normalized.append(parsed.model_dump())
                    except ValidationError as exc:
                        return [], f"Batch record validation failed at index {idx}: {exc}"
                else:
                    normalized.append(record)
            return normalized, None

        @tool(tool_name, description=description, args_schema=actual_args_schema)
        def ml_prediction_tool(
            return_probabilities: bool = default_return_probabilities,
            return_dict: bool = default_return_dict,
            batch_records: Optional[List[Dict[str, Any]]] = None,
            **features: Any,
        ) -> Any:
            model_record, resolved_path = resolve_model()
            if not resolved_path:
                return "Configured machine learning model is not available. Set model_identifier or model_file_path."

            resolved_path, path_error = self._ensure_local_model_path(model_record, resolved_path)
            if path_error:
                return path_error
            if not resolved_path:
                return "Model artifact could not be prepared for inference."

            try:
                input_schema: Optional[List[str]] = None
                active_record_model = record_model
                effective_feature_fields = feature_fields

                if model_record and active_record_model is None:
                    runtime_record_model, runtime_fields = self._build_record_model(model_record)
                    if runtime_record_model is not None and runtime_fields:
                        active_record_model = runtime_record_model
                        if feature_fields == ["record"]:
                            effective_feature_fields = runtime_fields

                if active_record_model is not None and effective_feature_fields != list(active_record_model.model_fields.keys()):
                    # Keep feature ordering consistent with validator whenever available.
                    effective_feature_fields = list(active_record_model.model_fields.keys())

                if batch_records:
                    normalized_batch, error_message = normalize_batch_records(
                        batch_records,
                        validator_model=active_record_model,
                    )
                    if error_message:
                        return error_message
                    if not normalized_batch:
                        return "At least one batch record is required for batch prediction."

                    inputs_payload: Any = normalized_batch
                    if active_record_model is not None:
                        input_schema = list(active_record_model.model_fields.keys())
                    elif effective_feature_fields and effective_feature_fields != ["record"]:
                        input_schema = effective_feature_fields
                    else:
                        sample = normalized_batch[0]
                        if isinstance(sample, dict):
                            input_schema = list(sample.keys())
                else:
                    if feature_fields == ["record"]:
                        record_payload = features.get("record")
                        if not isinstance(record_payload, dict):
                            return "Provide record as a dictionary that matches the model input schema."
                        if active_record_model is not None:
                            try:
                                parsed_record = active_record_model(**record_payload)
                                record_payload = parsed_record.model_dump()
                            except ValidationError as exc:
                                return f"Input validation failed: {exc}"
                            input_schema = list(active_record_model.model_fields.keys())
                        else:
                            input_schema = list(record_payload.keys())
                        inputs_payload = record_payload
                    else:
                        candidate_payload = {field: features.get(field) for field in feature_fields}
                        if active_record_model is not None:
                            try:
                                parsed_record = active_record_model(**candidate_payload)
                                record_payload = parsed_record.model_dump()
                            except ValidationError as exc:
                                return f"Input validation failed: {exc}"
                        else:
                            record_payload = candidate_payload

                        inputs_payload = record_payload
                        input_schema = effective_feature_fields

                inference_response = _inference_service.predict(
                    resolved_path,
                    inputs=inputs_payload,
                    input_schema=input_schema,
                    return_probabilities=return_probabilities,
                )
                output_schema = get_output_schema(model_record)
                enriched_response: Dict[str, Any]
                if isinstance(inference_response, dict):
                    enriched_response = dict(inference_response)
                else:
                    enriched_response = {"predictions": inference_response}

                if output_schema:
                    enriched_response.setdefault("metadata", {})
                    meta_block = enriched_response["metadata"]
                    if isinstance(meta_block, dict):
                        meta_block["output_schema"] = output_schema
                    else:
                        enriched_response["metadata"] = {"output_schema": output_schema}
                else:
                    enriched_response.setdefault("metadata", {})

                return self._format_response(enriched_response, return_dict)

            except FileNotFoundError:
                logger.error("Model artifact not found at path: %s", resolved_path)
                return f"Model artifact not found: {resolved_path}"
            except Exception as exc:  # pragma: no cover - inference level errors
                logger.exception("Machine learning prediction failed: %s", exc)
                return f"Machine learning prediction failed: {exc}"

        return ml_prediction_tool

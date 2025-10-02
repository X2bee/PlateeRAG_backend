"""Utilities for working with MLflow-backed model metadata."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from service.database.models.ml_model import MLModel


@dataclass(frozen=True)
class MlflowModelInfo:
    """Structured representation of an MLflow model reference."""

    model_uri: str
    tracking_uri: Optional[str] = None
    run_id: Optional[str] = None
    registered_model_name: Optional[str] = None
    model_version: Optional[str] = None
    load_flavor: str = "pyfunc"
    artifact_path: Optional[str] = None
    input_format: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def cache_key(self) -> str:
        tracking = self.tracking_uri or "default"
        return f"mlflow::{tracking}::{self.model_uri}"

    def cache_revision(self, model_record: Optional["MLModel"] = None) -> str:
        if model_record and getattr(model_record, "file_checksum", None):
            return str(model_record.file_checksum)
        if "cache_revision" in self.extra:
            return str(self.extra["cache_revision"])
        if model_record and getattr(model_record, "updated_at", None):
            return str(model_record.updated_at)
        return self.cache_key()


def _strip_none(meta: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in meta.items() if value is not None and value != ""}


def build_mlflow_metadata(
    *,
    tracking_uri: Optional[str],
    model_uri: str,
    run_id: Optional[str] = None,
    model_version: Optional[str] = None,
    registered_model_name: Optional[str] = None,
    load_flavor: Optional[str] = None,
    artifact_path: Optional[str] = None,
    input_format: Optional[str] = None,
    additional_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Builds a metadata payload containing MLflow reference information."""

    mlflow_block: Dict[str, Any] = _strip_none(
        {
            "tracking_uri": tracking_uri,
            "model_uri": model_uri,
            "run_id": run_id,
            "model_version": model_version,
            "registered_model_name": registered_model_name,
            "load_flavor": (load_flavor or "pyfunc").lower(),
            "artifact_path": artifact_path,
            "input_format": input_format,
        }
    )

    if additional_metadata:
        mlflow_block["additional_metadata"] = additional_metadata

    return {"mlflow": mlflow_block}


def merge_metadata(base: Optional[Dict[str, Any]], mlflow_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Merge MLflow metadata into an existing metadata dictionary."""

    merged: Dict[str, Any] = {}
    if base:
        merged.update(base)
    merged["mlflow"] = mlflow_metadata.get("mlflow", {})
    return merged


def compute_mlflow_checksum(
    *,
    tracking_uri: Optional[str],
    model_uri: str,
    run_id: Optional[str] = None,
    model_version: Optional[str] = None,
) -> str:
    """Generate a deterministic checksum for an MLflow model reference."""

    raw = "|".join(
        [
            "mlflow",
            tracking_uri or "",
            model_uri,
            run_id or "",
            model_version or "",
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def extract_mlflow_info(
    model_record: Optional["MLModel"] = None,
    file_path: Optional[str] = None,
) -> Optional[MlflowModelInfo]:
    """Extract MLflow reference information from a model record or file path."""

    inferred_uri = file_path or getattr(model_record, "file_path", None)

    if not inferred_uri:
        return None

    metadata_block: Dict[str, Any] = {}

    metadata = getattr(model_record, "metadata", None)
    if isinstance(metadata, dict):
        if isinstance(metadata.get("mlflow"), dict):
            metadata_block = metadata["mlflow"].copy()
        else:
            # Support older flat metadata formats
            flat_keys = {
                key: value
                for key, value in metadata.items()
                if key.startswith("mlflow_")
            }
            if flat_keys:
                metadata_block = {
                    "tracking_uri": flat_keys.get("mlflow_tracking_uri"),
                    "model_uri": flat_keys.get("mlflow_model_uri", inferred_uri),
                    "run_id": flat_keys.get("mlflow_run_id"),
                    "model_version": flat_keys.get("mlflow_model_version"),
                    "registered_model_name": flat_keys.get("mlflow_registered_model_name"),
                    "load_flavor": flat_keys.get("mlflow_load_flavor"),
                    "artifact_path": flat_keys.get("mlflow_artifact_path"),
                    "input_format": flat_keys.get("mlflow_input_format"),
                }
    framework = getattr(model_record, "framework", "")
    if not metadata_block and framework.lower() == "mlflow":
        metadata_block = {"model_uri": inferred_uri}

    if not metadata_block and isinstance(inferred_uri, str) and inferred_uri.startswith(("runs:/", "models:/", "models://", "mlflow:")):
        metadata_block = {"model_uri": inferred_uri}

    if not metadata_block or "model_uri" not in metadata_block:
        return None

    model_uri = metadata_block.get("model_uri", inferred_uri)
    tracking_uri = metadata_block.get("tracking_uri")
    run_id = metadata_block.get("run_id")
    model_version = metadata_block.get("model_version")
    registered_model_name = metadata_block.get("registered_model_name")
    load_flavor = metadata_block.get("load_flavor", "pyfunc")
    artifact_path = metadata_block.get("artifact_path")
    input_format = metadata_block.get("input_format")

    extra = {
        key: value
        for key, value in metadata_block.items()
        if key
        not in {
            "tracking_uri",
            "model_uri",
            "run_id",
            "model_version",
            "registered_model_name",
            "load_flavor",
            "artifact_path",
            "input_format",
        }
    }

    return MlflowModelInfo(
        model_uri=str(model_uri),
        tracking_uri=str(tracking_uri) if tracking_uri else None,
        run_id=str(run_id) if run_id else None,
        model_version=str(model_version) if model_version else None,
        registered_model_name=str(registered_model_name) if registered_model_name else None,
        load_flavor=str(load_flavor or "pyfunc"),
        artifact_path=str(artifact_path) if artifact_path else None,
        input_format=str(input_format) if input_format else None,
        extra=extra,
    )

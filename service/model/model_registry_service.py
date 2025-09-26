"""Service for managing the ML model registry."""
from __future__ import annotations

from typing import Optional, List, Dict, Any
from service.database.connection import AppDatabaseManager
from service.database.models.ml_model import MLModel
from service.model.mlflow_utils import (
    build_mlflow_metadata,
    compute_mlflow_checksum,
    merge_metadata,
)

class ModelRegistryService:
    """Provides CRUD helpers around the MLModel registry table."""

    def __init__(self, app_db: AppDatabaseManager):
        self.app_db = app_db

    def register_model(self, model_payload: Dict[str, Any]) -> MLModel:
        model = MLModel(**model_payload)
        insert_result = self.app_db.insert(model)
        if not insert_result or insert_result.get("result") != "success":
            raise RuntimeError("Failed to persist model metadata")

        model_id = insert_result.get("id")
        if model_id:
            created = self.app_db.find_by_id(MLModel, model_id)
            if created:
                return created

        # Fallback lookup when insert result lacks the identifier
        lookup = self.app_db.find_by_condition(
            MLModel,
            {"file_path": model.file_path},
            limit=1
        )
        if lookup:
            return lookup[0]

        return model

    def get_model_by_id(self, model_id: int) -> Optional[MLModel]:
        return self.app_db.find_by_id(MLModel, model_id)

    def get_model_by_name(self, model_name: str, model_version: Optional[str] = None) -> Optional[MLModel]:
        results = self.app_db.find_by_condition(
            MLModel,
            {"model_name": model_name},
            limit=25
        )
        if not results:
            return None
        if not model_version:
            return results[0]
        for record in results:
            if record.model_version == model_version:
                return record
        return None

    def get_model_by_checksum(self, checksum: str) -> Optional[MLModel]:
        results = self.app_db.find_by_condition(
            MLModel,
            {"file_checksum": checksum},
            limit=1
        )
        return results[0] if results else None

    def list_models(self, limit: int = 100, offset: int = 0) -> List[MLModel]:
        return self.app_db.find_all(MLModel, limit=limit, offset=offset)

    def update_model(self, model: MLModel) -> bool:
        result = self.app_db.update(model)
        return bool(result)

    def register_mlflow_model(
        self,
        *,
        model_name: str,
        model_uri: str,
        tracking_uri: Optional[str] = None,
        model_version: Optional[str] = None,
        description: Optional[str] = None,
        run_id: Optional[str] = None,
        registered_model_name: Optional[str] = None,
        load_flavor: Optional[str] = None,
        artifact_path: Optional[str] = None,
        input_format: Optional[str] = None,
        input_schema: Optional[List[str]] = None,
        output_schema: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        uploaded_by: Optional[int] = None,
        status: str = "active",
    ) -> MLModel:
        """Register a model that is stored and versioned by MLflow."""

        existing_mlflow_meta: Dict[str, Any] = {}
        user_metadata: Optional[Dict[str, Any]] = None
        if metadata and isinstance(metadata, dict):
            user_metadata = {key: value for key, value in metadata.items() if key != "mlflow"}
            raw_mlflow = metadata.get("mlflow") if isinstance(metadata, dict) else None
            if isinstance(raw_mlflow, dict):
                existing_mlflow_meta = raw_mlflow.copy()
        elif metadata:
            user_metadata = {"user_metadata": metadata}

        mlflow_metadata = build_mlflow_metadata(
            tracking_uri=tracking_uri,
            model_uri=model_uri,
            run_id=run_id,
            model_version=model_version,
            registered_model_name=registered_model_name,
            load_flavor=load_flavor,
            artifact_path=artifact_path,
            input_format=input_format,
            additional_metadata=existing_mlflow_meta.get("additional_metadata"),
        )

        merged_mlflow_block = mlflow_metadata.get("mlflow", {})
        for key, value in existing_mlflow_meta.items():
            if key not in merged_mlflow_block and key != "additional_metadata":
                merged_mlflow_block[key] = value

        combined_metadata = merge_metadata(user_metadata, {"mlflow": merged_mlflow_block})

        checksum = compute_mlflow_checksum(
            tracking_uri=tracking_uri,
            model_uri=model_uri,
            run_id=run_id,
            model_version=model_version,
        )

        payload: Dict[str, Any] = {
            "model_name": model_name,
            "model_version": model_version,
            "framework": "mlflow",
            "description": description,
            "file_path": model_uri,
            "file_size": None,
            "file_checksum": checksum,
            "input_schema": input_schema,
            "output_schema": output_schema,
            "metadata": combined_metadata,
            "uploaded_by": uploaded_by,
            "status": status,
        }

        return self.register_model(payload)

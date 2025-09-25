"""Service for managing the ML model registry."""
from __future__ import annotations

from typing import Optional, List, Dict, Any
from service.database.connection import AppDatabaseManager
from service.database.models.ml_model import MLModel

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

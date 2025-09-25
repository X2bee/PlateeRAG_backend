"""Service helpers for deleting registered ML models."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from service.database.connection import AppDatabaseManager
from service.database.models.ml_model import MLModel

logger = logging.getLogger("model-deletion-service")


class ModelDeletionService:
    """Encapsulates model lookup and deletion logic."""

    def __init__(self, app_db: AppDatabaseManager):
        self.app_db = app_db

    def get_by_id(self, model_id: int) -> Optional[MLModel]:
        return self.app_db.find_by_id(MLModel, model_id)

    def delete(
        self,
        model_id: int,
        *,
        model: Optional[MLModel] = None,
        remove_artifact: bool = True,
    ) -> bool:
        target = model or self.get_by_id(model_id)
        if not target:
            return False

        if remove_artifact:
            file_path = getattr(target, "file_path", None)
            if file_path:
                try:
                    path = Path(file_path).expanduser()
                except (TypeError, ValueError):
                    path = None

                if path and path.exists():
                    try:
                        path.unlink()
                        logger.info("Removed model artifact: %s", path)
                    except OSError as exc:
                        logger.error("Failed to remove artifact %s: %s", path, exc)
                        raise RuntimeError("MODEL_ARTIFACT_DELETE_FAILED") from exc

        return self.app_db.delete(MLModel, target.id)

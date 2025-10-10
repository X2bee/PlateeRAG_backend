"""Service helpers for deleting registered ML models."""
from __future__ import annotations

import importlib
import logging
from typing import Optional

from service.database.connection import AppDatabaseManager
from service.database.models.ml_model import MLModel
from service.model.mlflow_utils import MlflowModelInfo, extract_mlflow_info

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
            mlflow_info = extract_mlflow_info(target)
            if not mlflow_info:
                logger.warning(
                    "MLflow metadata missing for model %s; skipping remote artifact deletion",
                    target.id,
                )
            else:
                try:
                    self._delete_mlflow_artifact(mlflow_info)
                except RuntimeError:
                    raise
                except Exception as exc:  # pragma: no cover - best effort remote cleanup
                    logger.error("Failed to remove MLflow artifact: %s", exc, exc_info=True)
                    raise RuntimeError("MODEL_ARTIFACT_DELETE_FAILED") from exc

        return self.app_db.delete(MLModel, target.id)

    def _delete_mlflow_artifact(self, info: MlflowModelInfo) -> None:
        self._require_mlflow()

        tracking_kwargs = {}
        if info.tracking_uri:
            tracking_kwargs["tracking_uri"] = info.tracking_uri

        try:
            tracking_module = importlib.import_module("mlflow.tracking")
            client_cls = getattr(tracking_module, "MlflowClient")
        except ImportError as exc:  # pragma: no cover - optional dependency missing
            raise RuntimeError("MLflow tracking client unavailable") from exc
        except AttributeError as exc:  # pragma: no cover - defensive
            raise RuntimeError("MlflowClient class not found") from exc

        client = client_cls(**tracking_kwargs)

        if info.registered_model_name and info.model_version:
            client.delete_model_version(info.registered_model_name, str(info.model_version))
            logger.info(
                "Deleted MLflow model version %s (model=%s tracking=%s)",
                info.model_version,
                info.registered_model_name,
                info.tracking_uri,
            )
            return

        delete_run_completely = bool(info.extra.get("delete_run_on_remove")) if info.extra else False
        delete_run_artifacts = bool(info.extra.get("delete_run_artifacts")) if info.extra else False

        if info.run_id:
            if delete_run_completely:
                client.delete_run(info.run_id)
                logger.info(
                    "Deleted MLflow run %s from tracking server %s",
                    info.run_id,
                    info.tracking_uri,
                )
                return

            artifact_path = info.artifact_path or info.extra.get("artifact_path") if info.extra else None
            if artifact_path:
                client.delete_run_artifacts(info.run_id, artifact_path)
                logger.info(
                    "Deleted MLflow run artifacts path '%s' for run %s",
                    artifact_path,
                    info.run_id,
                )
                return

            if delete_run_artifacts:
                client.delete_run_artifacts(info.run_id)
                logger.info(
                    "Deleted all MLflow run artifacts for run %s",
                    info.run_id,
                )
                return

        logger.info(
            "No MLflow artifact cleanup performed for model %s (tracking URI: %s)",
            info.model_uri,
            info.tracking_uri,
        )

    @staticmethod
    def _require_mlflow():
        try:
            return importlib.import_module("mlflow")
        except ImportError as exc:  # pragma: no cover - optional dependency missing
            raise RuntimeError("MLflow support requires the 'mlflow' package") from exc

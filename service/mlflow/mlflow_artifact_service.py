"""Utilities for interacting with MLflow artifacts."""
from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

LOGGER = logging.getLogger("mlflow-artifact-service")

try:  # Optional dependency managed via pyproject
    import mlflow  # type: ignore
    from mlflow.exceptions import MlflowException  # type: ignore
    from mlflow.tracking import MlflowClient  # type: ignore
except ImportError as exc:  # pragma: no cover - handled at runtime
    mlflow = None
    MlflowClient = None
    MlflowException = Exception  # type: ignore
    _IMPORT_ERROR: Optional[ImportError] = exc
else:  # pragma: no branch
    _IMPORT_ERROR = None


ARTIFACT_URI_PREFIX = "mlflow://"


class MLflowArtifactService:
    """High level helper that wraps :class:`MlflowClient`.

    The service keeps a small on-disk cache for downloaded artifacts so that
    repeated inferences can reuse the same serialized model without contacting
    the MLflow server every time.
    """

    def __init__(
        self,
        tracking_uri: str,
        *,
        default_experiment_id: Optional[str] = None,
        cache_dir: Optional[Path | str] = None,
        tracking_token: Optional[str] = None,
    ) -> None:
        if mlflow is None or MlflowClient is None:  # pragma: no cover - import guard
            raise RuntimeError(
                "mlflow package is not available. Install project dependencies first"
            ) from _IMPORT_ERROR

        self.tracking_uri = tracking_uri.rstrip("/")
        self.default_experiment_id = default_experiment_id
        self.cache_dir = Path(cache_dir or Path.cwd() / ".mlflow_cache").resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if tracking_token:
            os.environ.setdefault("MLFLOW_TRACKING_TOKEN", tracking_token)

        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = MlflowClient(tracking_uri=self.tracking_uri)

    # ------------------------------------------------------------------
    # Artifact URI helpers
    # ------------------------------------------------------------------
    @staticmethod
    def build_artifact_uri(run_id: str, artifact_path: str) -> str:
        artifact_path = artifact_path.lstrip("/")
        return f"{ARTIFACT_URI_PREFIX}{run_id}/{artifact_path}" if artifact_path else f"{ARTIFACT_URI_PREFIX}{run_id}"

    @staticmethod
    def parse_artifact_uri(uri: str) -> Tuple[str, str]:
        if not uri.startswith(ARTIFACT_URI_PREFIX):
            raise ValueError(f"Unsupported artifact URI: {uri}")
        remainder = uri[len(ARTIFACT_URI_PREFIX):]
        if "/" in remainder:
            run_id, artifact_path = remainder.split("/", 1)
        else:
            run_id, artifact_path = remainder, ""
        return run_id, artifact_path

    # ------------------------------------------------------------------
    # Run helpers
    # ------------------------------------------------------------------
    def ensure_run(
        self,
        *,
        run_id: Optional[str] = None,
        experiment_id: Optional[str] = None,
        run_name: Optional[str] = None,
    ) -> str:
        if run_id:
            return run_id

        experiment = experiment_id or self.default_experiment_id
        if not experiment:
            raise ValueError("experiment_id is required when run_id is not provided")

        tags = {"mlflow.runName": run_name} if run_name else None
        run = self.client.create_run(experiment_id=str(experiment), tags=tags)
        LOGGER.info("Created MLflow run %s under experiment %s", run.info.run_id, experiment)
        return run.info.run_id

    # ------------------------------------------------------------------
    # Artifact operations
    # ------------------------------------------------------------------
    def log_artifact(
        self,
        run_id: str,
        local_path: Path,
        *,
        artifact_dir: Optional[str] = None,
    ) -> str:
        artifact_dir_normalized = (artifact_dir or "").strip("/")
        artifact_dir_arg = artifact_dir_normalized or None

        self.client.log_artifact(run_id, str(local_path), artifact_path=artifact_dir_arg)

        artifact_relative_path = (
            f"{artifact_dir_normalized}/{local_path.name}" if artifact_dir_normalized else local_path.name
        )
        LOGGER.info(
            "Uploaded artifact to MLflow | run_id=%s artifact_path=%s",
            run_id,
            artifact_relative_path,
        )
        return artifact_relative_path

    def list_artifacts(self, run_id: str, path: Optional[str] = None) -> Iterable[Dict[str, Any]]:
        items = self.client.list_artifacts(run_id, path.strip("/") or None)
        for entry in items:
            yield {
                "path": entry.path,
                "is_dir": entry.is_dir,
                "file_size": getattr(entry, "file_size", None),
            }

    def delete_artifact(self, run_id: str, artifact_path: str) -> None:
        artifact_path = artifact_path.strip("/")
        try:
            self.client.delete_artifacts(run_id, artifact_path or None)
            LOGGER.info(
                "Deleted MLflow artifact | run_id=%s artifact_path=%s",
                run_id,
                artifact_path,
            )
        except MlflowException as exc:  # pragma: no cover - depends on server response
            LOGGER.warning(
                "Failed to delete MLflow artifact | run_id=%s artifact_path=%s error=%s",
                run_id,
                artifact_path,
                exc,
            )
            raise

    def artifact_exists(self, run_id: str, artifact_path: str) -> bool:
        artifact_path = artifact_path.strip("/")
        parent, _, name = artifact_path.rpartition("/")
        parent = parent or None
        try:
            for entry in self.client.list_artifacts(run_id, parent):
                if entry.path.strip("/") == artifact_path:
                    return True
        except MlflowException:
            return False
        return False

    # ------------------------------------------------------------------
    # Model registry helpers
    # ------------------------------------------------------------------
    def list_registered_models(self, max_results: int = 200) -> List[Any]:
        """Return registered models from MLflow."""
        # search_registered_models already supports pagination; simple loop covers all pages.
        page_token: Optional[str] = None
        collected: List[Any] = []
        while True:
            response = self.client.search_registered_models(page_token=page_token, max_results=max_results)
            collected.extend(response)
            page_token = getattr(response, "token", None) or getattr(response, "next_page_token", None)
            if not page_token:
                break
        return collected

    def list_model_versions(self, model_name: str, max_results: int = 200) -> List[Any]:
        """Return all versions for a registered model."""
        filter_string = f"name='{model_name}'"
        page_token: Optional[str] = None
        collected: List[Any] = []
        while True:
            response = self.client.search_model_versions(filter_string, page_token=page_token, max_results=max_results)
            collected.extend(response)
            page_token = getattr(response, "token", None) or getattr(response, "next_page_token", None)
            if not page_token:
                break
        return collected

    # ------------------------------------------------------------------
    # Local cache helpers
    # ------------------------------------------------------------------
    def ensure_local_artifact(
        self,
        artifact_uri: str,
        *,
        expected_checksum: Optional[str] = None,
    ) -> Path:
        run_id, artifact_path = self.parse_artifact_uri(artifact_uri)
        target = self.cache_dir / run_id / artifact_path
        target.parent.mkdir(parents=True, exist_ok=True)

        if target.exists():
            if expected_checksum and self._compute_checksum(target) != expected_checksum:
                LOGGER.warning(
                    "Cached artifact checksum mismatch; redownloading | uri=%s", artifact_uri
                )
                target.unlink(missing_ok=True)
            else:
                return target

        downloaded_path = Path(
            self.client.download_artifacts(run_id, artifact_path, dst_path=str(target.parent))
        )

        if downloaded_path.is_dir():  # pragma: no cover - defensive guard
            raise ValueError(
                f"Artifact path '{artifact_path}' refers to a directory. Expected a file."
            )

        if downloaded_path != target:
            downloaded_path.replace(target)

        if expected_checksum:
            checksum = self._compute_checksum(target)
            if checksum != expected_checksum:
                target.unlink(missing_ok=True)
                raise ValueError(
                    "Downloaded MLflow artifact checksum mismatch. Expected "
                    f"{expected_checksum}, got {checksum}."
                )

        return target

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_checksum(path: Path) -> str:
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

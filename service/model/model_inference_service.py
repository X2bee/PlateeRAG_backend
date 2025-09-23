"""Utility for loading models and running inference."""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

try:  # Optional dependency for serialization
    import joblib  # type: ignore
except ImportError:  # pragma: no cover - fallback handled below
    joblib = None

import pickle

try:  # Optional scientific stack
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover - fallback handled below
    np = None

logger = logging.getLogger("model-inference-service")

class ModelInferenceService:
    """Provides cached model loading and inference helpers."""

    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()

    def load_model(self, file_path: str) -> Any:
        path = Path(file_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found: {path}")

        with self._lock:
            cache_entry = self._cache.get(str(path))
            mtime = path.stat().st_mtime
            if cache_entry and cache_entry.get("mtime") == mtime:
                return cache_entry["model"]

            if joblib is not None:
                model = joblib.load(path)
            else:
                with path.open("rb") as artifact:
                    model = pickle.load(artifact)
            self._cache[str(path)] = {"model": model, "mtime": mtime}
            logger.info("Loaded model from %s", path)
            return model

    @staticmethod
    def _normalize_inputs(raw_inputs: Any, input_schema: Optional[List[str]] = None) -> Any:
        if raw_inputs is None:
            raise ValueError("Inputs are required for inference")

        if isinstance(raw_inputs, list) and not raw_inputs:
            raise ValueError("At least one record is required for inference")

        # Single record passed as dictionary
        if isinstance(raw_inputs, dict):
            if not input_schema:
                raise ValueError("Input schema required when passing a dictionary record")
            row = [raw_inputs[field] for field in input_schema]
            return ModelInferenceService._to_array([row])

        if not isinstance(raw_inputs, list):
            raise ValueError("Inputs must be a list, list of lists, or list of dicts")

        first_item = raw_inputs[0]

        if isinstance(first_item, dict):
            if not input_schema:
                raise ValueError("Input schema required when passing dictionaries")
            rows: List[List[Any]] = []
            for idx, item in enumerate(raw_inputs):
                row = []
                for field in input_schema:
                    if field not in item:
                        raise ValueError(f"Missing feature '{field}' in record index {idx}")
                    row.append(item[field])
                rows.append(row)
            return ModelInferenceService._to_array(rows)

        if isinstance(first_item, (list, tuple)):
            rows = [list(item) for item in raw_inputs]
            return ModelInferenceService._to_array(rows)

        # Treat as a single record represented by scalars
        return ModelInferenceService._to_array([raw_inputs])

    def predict(self,
                file_path: str,
                inputs: Any,
                input_schema: Optional[List[str]] = None,
                return_probabilities: bool = False) -> Dict[str, Any]:
        model = self.load_model(file_path)
        normalized_inputs = self._normalize_inputs(inputs, input_schema)

        predictions = model.predict(normalized_inputs)
        response: Dict[str, Any] = {
            "predictions": self._to_serializable(predictions)
        }

        if return_probabilities and hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(normalized_inputs)
            response["probabilities"] = self._to_serializable(probabilities)

        return response

    @staticmethod
    def _to_serializable(value: Any) -> Any:
        if np is not None and isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (list, tuple)):
            return [ModelInferenceService._to_serializable(item) for item in value]
        if hasattr(value, "tolist"):
            try:
                return value.tolist()
            except Exception:  # pragma: no cover - fallback handler
                pass
        return value

    @staticmethod
    def _to_array(rows: List[List[Any]]) -> Any:
        if np is not None:
            return np.array(rows, dtype=object)
        return [list(row) for row in rows]

    def clear_cache(self, file_path: str):
        resolved = str(Path(file_path).expanduser().resolve())
        with self._lock:
            if resolved in self._cache:
                del self._cache[resolved]

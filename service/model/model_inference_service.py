"""Utility for loading models and running inference."""
from __future__ import annotations

import contextlib
import functools
import importlib
import inspect
import logging
import json
import pickle
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

try:  # Optional dependency for serialization
    import joblib  # type: ignore
except ImportError:  # pragma: no cover - fallback handled below
    joblib = None

try:  # Optional scientific stack
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover - fallback handled below
    np = None

from service.model.mlflow_utils import MlflowModelInfo, extract_mlflow_info

if TYPE_CHECKING:  # pragma: no cover - used for typing only
    from service.database.models.ml_model import MLModel

logger = logging.getLogger("model-inference-service")

MODULE_FALLBACKS = {
    "tests.assets.sample_demo_model": "service.model.sample_demo_model",
}


@dataclass(frozen=True)
class _ModelSource:
    """Internal representation of a model's physical location."""

    kind: str
    cache_key: str
    cache_revision: Optional[Any]
    path: Optional[Path] = None
    mlflow_info: Optional[MlflowModelInfo] = None
    original_uri: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class ModelInferenceService:
    """Provides cached model loading and inference helpers."""

    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()

    def load_model(self, file_path: str, *, model_record: Optional["MLModel"] = None) -> Any:
        source = self._resolve_source(file_path, model_record)
        logger.info("[inference-service] Loading model | kind=%s cache_key=%s", source.kind, source.cache_key)
        if source.kind == "mlflow":
            return self._load_mlflow_model(source)
        return self._load_local_model(source)

    def _resolve_source(
        self,
        file_path: str,
        model_record: Optional["MLModel"],
    ) -> _ModelSource:
        mlflow_info = extract_mlflow_info(model_record, file_path)
        metadata_context = self._build_metadata_context(model_record, mlflow_info)
        if mlflow_info:
            context_copy = metadata_context.copy()
            return _ModelSource(
                kind="mlflow",
                cache_key=mlflow_info.cache_key(),
                cache_revision=mlflow_info.cache_revision(model_record),
                mlflow_info=mlflow_info,
                original_uri=file_path,
                metadata=context_copy,
            )

        path = Path(file_path).expanduser().resolve()
        context_copy = metadata_context.copy()
        return _ModelSource(
            kind="local",
            cache_key=str(path),
            cache_revision=None,
            path=path,
            original_uri=str(path),
            metadata=context_copy,
        )

    @staticmethod
    def _coerce_metadata_dict(value: Any) -> Optional[Dict[str, Any]]:
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except (TypeError, ValueError):
                return None
            if isinstance(parsed, dict):
                return parsed
        return None

    @classmethod
    def _extract_model_tags(cls, model_record: Optional["MLModel"]) -> Dict[str, Any]:
        if model_record is None:
            return {}
        metadata_dict = cls._coerce_metadata_dict(getattr(model_record, "metadata", None))
        if not isinstance(metadata_dict, dict):
            return {}
        mlflow_meta = metadata_dict.get("mlflow")
        if not isinstance(mlflow_meta, dict):
            return {}
        additional = cls._coerce_metadata_dict(mlflow_meta.get("additional_metadata"))
        if not isinstance(additional, dict):
            return {}
        tags = additional.get("tags")
        if isinstance(tags, dict):
            return tags
        if isinstance(tags, str):
            try:
                parsed = json.loads(tags)
            except (TypeError, ValueError):
                return {}
            return parsed if isinstance(parsed, dict) else {}
        return {}

    @classmethod
    def _build_metadata_context(
        cls,
        model_record: Optional["MLModel"],
        mlflow_info: Optional[MlflowModelInfo],
    ) -> Dict[str, Any]:
        context: Dict[str, Any] = {}
        if mlflow_info and mlflow_info.extra:
            context.update(mlflow_info.extra)
        tags = cls._extract_model_tags(model_record)
        if tags:
            existing_tags = context.get("tags")
            if isinstance(existing_tags, dict):
                merged = existing_tags.copy()
                merged.update({k: v for k, v in tags.items() if k not in merged})
                context["tags"] = merged
            else:
                context["tags"] = tags
        return context

    @staticmethod
    def _extract_flavor_hint(metadata_context: Dict[str, Any]) -> Optional[str]:
        keys = (
            "mlflow_load_flavor",
            "user_script_model_primary_flavor",
            "model_format",
            "xgenml_model_format",
        )

        candidates: List[Any] = []
        tags = metadata_context.get("tags")
        if isinstance(tags, dict):
            candidates.extend(tags.get(key) for key in keys)

        for key in keys:
            if key in metadata_context:
                candidates.append(metadata_context[key])

        for value in candidates:
            if isinstance(value, str) and value.strip():
                return value.strip().lower()
        return None

    def _load_local_model(self, source: _ModelSource) -> Any:
        path = source.path
        if not path or not path.exists():
            raise FileNotFoundError(f"Model artifact not found: {source.original_uri}")

        mtime = path.stat().st_mtime

        with self._lock:
            cache_entry = self._cache.get(source.cache_key)
            if cache_entry and cache_entry.get("revision") == mtime:
                return cache_entry["model"]

        script_path_added: Optional[str] = None
        script_candidate: Optional[Path] = None
        if path.is_dir():
            candidate = path / "user_script.py"
            if candidate.exists():
                script_candidate = candidate
        else:
            candidate = path.parent / "user_script.py"
            if candidate.exists():
                script_candidate = candidate

        if script_candidate:
            script_dir = str(script_candidate.parent)
            if script_dir not in sys.path:
                sys.path.insert(0, script_dir)
                script_path_added = script_dir
                logger.info("Temporarily added user script directory to sys.path: %s", script_dir)

        flavor_hint = self._extract_flavor_hint(source.metadata)

        try:
            model = self._load_artifact(path, flavor_hint=flavor_hint)
        except ModuleNotFoundError as exc:
            missing_module = getattr(exc, "name", None)
            if missing_module and self._ensure_fallback_module(missing_module):
                model = self._load_artifact(path, flavor_hint=flavor_hint)
            else:
                raise
        finally:
            if script_path_added:
                try:
                    sys.path.remove(script_path_added)
                except ValueError:
                    pass

        with self._lock:
            self._cache[source.cache_key] = {"model": model, "revision": mtime}

        logger.info("Loaded model from %s", path)
        return model

    def _load_mlflow_model(self, source: _ModelSource) -> Any:
        info = source.mlflow_info
        if info is None:
            raise RuntimeError("MLflow model metadata missing for remote model load")

        logger.info(
            "[inference-service] Preparing MLflow load | model_uri=%s run_id=%s artifact_path=%s",
            info.model_uri,
            info.run_id,
            info.artifact_path,
        )
        revision_key = source.cache_revision or source.cache_key

        with self._lock:
            cache_entry = self._cache.get(source.cache_key)
            if cache_entry and cache_entry.get("revision") == revision_key:
                return cache_entry["model"]

        model = self._fetch_mlflow_model(info)

        with self._lock:
            self._cache[source.cache_key] = {
                "model": model,
                "revision": revision_key,
            }

        logger.info(
            "[inference-service] Loaded MLflow model | uri=%s tracking_uri=%s",
            info.model_uri,
            info.tracking_uri,
        )
        return model

    def _fetch_mlflow_model(self, info: MlflowModelInfo) -> Any:
        mlflow_module = self._require_mlflow()
        loader = self._select_mlflow_loader(mlflow_module, info.load_flavor)
        load_kwargs = self._resolve_mlflow_load_kwargs(loader, info)

        logger.info(
            "[inference-service] Invoking MLflow loader | flavor=%s kwargs=%s",
            info.load_flavor,
            load_kwargs,
        )
        with self._mlflow_tracking_context(mlflow_module, info.tracking_uri):
            return loader(info.model_uri, **load_kwargs)

    @staticmethod
    def _resolve_mlflow_load_kwargs(loader: Any, info: MlflowModelInfo) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        if info.extra:
            extra_kwargs = info.extra.get("load_kwargs")
            if isinstance(extra_kwargs, dict):
                kwargs.update(extra_kwargs)
            tags = info.extra.get("tags")
            tag_flavor = None
            if isinstance(tags, dict):
                tag_flavor = tags.get("mlflow_load_flavor") or tags.get("user_script_model_primary_flavor")
            if tag_flavor and isinstance(tag_flavor, str):
                tag_flavor = tag_flavor.lower()
            if tag_flavor == "torchscript":
                kwargs.setdefault("map_location", "cpu")

        try:
            signature = inspect.signature(loader)
        except (TypeError, ValueError):  # pragma: no cover - loader without signature
            signature = None

        if signature:
            parameters = signature.parameters
            if info.run_id and "run_id" in parameters and "run_id" not in kwargs:
                kwargs["run_id"] = info.run_id
            if info.artifact_path and "artifact_path" in parameters and "artifact_path" not in kwargs:
                kwargs["artifact_path"] = info.artifact_path
            if "map_location" not in parameters and "map_location" in kwargs:
                kwargs.pop("map_location", None)

        return kwargs

    @staticmethod
    def _select_mlflow_loader(mlflow_module: Any, flavor: str):
        flavor_normalized = (flavor or "pyfunc").lower()
        alias_map = {
            "torchscript": "pytorch",
        }
        target_attr = alias_map.get(flavor_normalized, flavor_normalized)
        if target_attr == "pyfunc":
            target_attr = "pyfunc"
        namespace = getattr(mlflow_module, target_attr, None)
        if namespace is None or not hasattr(namespace, "load_model"):
            raise RuntimeError(
                f"MLflow flavor '{flavor}' is not available. Ensure the appropriate MLflow extra is installed."
            )
        return namespace.load_model

    @staticmethod
    def _require_mlflow():
        try:
            return importlib.import_module("mlflow")
        except ImportError as exc:  # pragma: no cover - depends on optional dependency
            raise RuntimeError(
                "MLflow support requires the 'mlflow' package to be installed."
            ) from exc

    @contextlib.contextmanager
    def _mlflow_tracking_context(self, mlflow_module: Any, tracking_uri: Optional[str]):
        if not tracking_uri:
            yield
            return

        set_tracking = getattr(mlflow_module, "set_tracking_uri", None)
        get_tracking = getattr(mlflow_module, "get_tracking_uri", None)

        if not callable(set_tracking):
            yield
            return

        previous_uri = None
        if callable(get_tracking):
            try:
                previous_uri = get_tracking()
            except Exception:  # pragma: no cover - defensive fallback
                previous_uri = None

        try:
            set_tracking(tracking_uri)
            yield
        finally:
            if previous_uri is not None:
                try:
                    set_tracking(previous_uri)
                except Exception:  # pragma: no cover - best effort restoration
                    logger.warning("Failed to restore MLflow tracking URI", exc_info=True)

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
            row = [ModelInferenceService._coerce_value(raw_inputs[field]) for field in input_schema]
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
                    row.append(ModelInferenceService._coerce_value(item[field]))
                rows.append(row)
            return ModelInferenceService._to_array(rows)

        if isinstance(first_item, (list, tuple)):
            rows = [
                [ModelInferenceService._coerce_value(value) for value in list(item)]
                for item in raw_inputs
            ]
            return ModelInferenceService._to_array(rows)

        if isinstance(raw_inputs, (list, tuple)):
            coerced = [ModelInferenceService._coerce_value(value) for value in raw_inputs]
        else:
            coerced = [ModelInferenceService._coerce_value(raw_inputs)]
        return ModelInferenceService._to_array([coerced])

    def _prepare_inputs_for_model(
        self,
        normalized_inputs: Any,
        input_schema: Optional[List[str]],
        mlflow_info: Optional[MlflowModelInfo],
    ) -> Any:
        if not mlflow_info:
            return normalized_inputs

        preferred_format = (mlflow_info.input_format or "").lower()
        if not preferred_format and mlflow_info.extra:
            preferred_format = str(mlflow_info.extra.get("input_format", "")).lower()

        expects_dataframe = preferred_format in {"dataframe", "df", "pandas"}
        if not expects_dataframe and mlflow_info.extra:
            expects_dataframe = bool(mlflow_info.extra.get("expect_dataframe"))

        if not expects_dataframe:
            return normalized_inputs

        try:
            pandas = importlib.import_module("pandas")
        except ImportError:  # pragma: no cover - optional dependency
            logger.warning(
                "Pandas is required to convert inputs to DataFrame for this MLflow model. "
                "Continuing with array inputs."
            )
            return normalized_inputs

        rows = normalized_inputs.tolist() if hasattr(normalized_inputs, "tolist") else normalized_inputs
        if not isinstance(rows, list):
            return normalized_inputs

        columns: Optional[List[str]] = input_schema
        if not columns and mlflow_info.extra:
            maybe_columns = mlflow_info.extra.get("columns")
            if isinstance(maybe_columns, list):
                columns = [str(col) for col in maybe_columns]

        try:
            if columns:
                return pandas.DataFrame(rows, columns=columns)
            return pandas.DataFrame(rows)
        except Exception:  # pragma: no cover - pandas DataFrame construction failure
            logger.warning("Failed to convert inputs to pandas.DataFrame", exc_info=True)
            return normalized_inputs

    def predict(
        self,
        file_path: str,
        inputs: Any,
        input_schema: Optional[List[str]] = None,
        return_probabilities: bool = False,
        model_record: Optional["MLModel"] = None,
    ) -> Dict[str, Any]:
        mlflow_info = extract_mlflow_info(model_record, file_path)
        model = self.load_model(file_path, model_record=model_record)
        normalized_inputs = self._normalize_inputs(inputs, input_schema)
        prepared_inputs = self._prepare_inputs_for_model(normalized_inputs, input_schema, mlflow_info)

        is_torch_model = self._is_torch_model(model)

        if is_torch_model:
            predictions = self._predict_with_torch(model, prepared_inputs)
        else:
            if not hasattr(model, "predict"):
                raise AttributeError("Loaded model does not expose a predict method")
            predictions = model.predict(prepared_inputs)

        response: Dict[str, Any] = {"predictions": self._to_serializable(predictions)}

        if return_probabilities:
            if is_torch_model:
                logger.warning("Probability outputs are not supported for torch-based user script models.")
            elif hasattr(model, "predict_proba"):
                try:
                    probabilities = model.predict_proba(prepared_inputs)
                except Exception as exc:  # pragma: no cover - model specific failure
                    logger.warning("Model does not support probability prediction: %s", exc)
                else:
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

    @staticmethod
    def _coerce_value(value: Any) -> Any:
        if isinstance(value, str):
            stripped = value.strip()
            if stripped == "":
                return value
            try:
                return float(stripped)
            except ValueError:
                return value
        return value

    @staticmethod
    def _is_torch_artifact_path(path: Path) -> bool:
        return path.suffix.lower() in {".pt", ".pth", ".torch"}

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def _import_torch():
        try:
            import torch  # type: ignore
        except ImportError:
            return None
        return torch

    @staticmethod
    def _is_torch_model(model: Any) -> bool:
        torch = ModelInferenceService._import_torch()
        if torch is None:
            return False

        script_module_cls = getattr(torch.jit, "ScriptModule", tuple())
        module_types = tuple(
            cls for cls in (getattr(torch.nn, "Module", None), script_module_cls) if cls
        )
        if not module_types:
            return False

        return isinstance(model, module_types)

    @staticmethod
    def _predict_with_torch(model: Any, prepared_inputs: Any) -> Any:
        torch = ModelInferenceService._import_torch()
        if torch is None:
            raise RuntimeError("PyTorch runtime is required to run torch-based models.")

        if hasattr(model, "eval"):
            model.eval()

        if isinstance(prepared_inputs, list):
            array_like = prepared_inputs
        elif hasattr(prepared_inputs, "to_numpy"):
            array_like = prepared_inputs.to_numpy()
        elif np is not None:
            array_like = np.asarray(prepared_inputs)
        else:
            array_like = prepared_inputs

        if np is not None:
            array_like = np.asarray(array_like)
            if array_like.dtype.kind not in {"f", "c"}:
                array_like = array_like.astype("float32")

        tensor = torch.as_tensor(array_like, dtype=torch.float32)
        with torch.no_grad():
            output = model(tensor)

        if hasattr(output, "detach"):
            output = output.detach()
        if hasattr(output, "cpu"):
            output = output.cpu()
        if hasattr(output, "numpy"):
            output = output.numpy()

        return output

    @staticmethod
    def _load_artifact(path: Path, flavor_hint: Optional[str] = None) -> Any:
        flavor = (flavor_hint or "").lower()
        use_torch = False
        prefer_torchscript = False

        if flavor in {"torchscript", "pytorch"}:
            use_torch = True
            prefer_torchscript = flavor == "torchscript"
        elif ModelInferenceService._is_torch_artifact_path(path):
            use_torch = True
            prefer_torchscript = True

        if use_torch:
            torch = ModelInferenceService._import_torch()
            if torch is None:
                raise RuntimeError(
                    "PyTorch runtime is required to load torch-based user script models."
                )
            model = None
            if prefer_torchscript:
                try:
                    model = torch.jit.load(str(path), map_location="cpu")
                except (RuntimeError, ValueError) as exc:
                    logger.warning(
                        "Failed to load TorchScript artifact at %s: %s. Falling back to torch.load.",
                        path,
                        exc,
                    )
            if model is None:
                model = torch.load(str(path), map_location="cpu")
            if hasattr(model, "eval"):
                model.eval()
            return model

        if joblib is not None:
            return joblib.load(path)
        with path.open("rb") as artifact:
            return pickle.load(artifact)

    def _ensure_fallback_module(self, module_name: str) -> bool:
        fallback = MODULE_FALLBACKS.get(module_name)
        if not fallback:
            return False
        try:
            module = importlib.import_module(fallback)
            sys.modules[module_name] = module
            logger.warning(
                "Registered fallback module mapping %s -> %s", module_name, fallback
            )
            return True
        except ModuleNotFoundError:
            logger.error("Fallback module %s not found", fallback)
            return False

    def clear_cache(self, file_path: str, *, model_record: Optional["MLModel"] = None):
        source = self._resolve_source(file_path, model_record)
        with self._lock:
            if source.cache_key in self._cache:
                del self._cache[source.cache_key]

"""Fallback DemoClassifier matching the test sample artifact."""
from __future__ import annotations

try:  # optional
    import numpy as _np  # type: ignore
except Exception:  # pragma: no cover
    _np = None


class DemoClassifier:
    def __init__(self):
        self.class_count = 3

    def predict(self, data):
        rows = self._normalize(data)
        predictions = []
        for row in rows:
            feature_c = row[2] if len(row) > 2 else 0
            feature_a = row[0] if len(row) > 0 else 0
            if feature_c > 3:
                predictions.append(2)
            elif feature_a > 5.5:
                predictions.append(1)
            else:
                predictions.append(0)
        return predictions

    def predict_proba(self, data):
        preds = self.predict(data)
        proba = []
        for pred in preds:
            dist = [0.0, 0.0, 0.0]
            dist[int(pred)] = 1.0
            proba.append(dist)
        return proba

    def _normalize(self, data):
        if _np is not None and isinstance(data, _np.ndarray):
            data = data.tolist()
        elif hasattr(data, "tolist") and not isinstance(data, list):
            data = data.tolist()

        if not isinstance(data, list):
            raise TypeError("Data must be list-like")
        normalized = []
        for item in data:
            if isinstance(item, dict):
                normalized.append([item[key] for key in sorted(item.keys())])
            elif isinstance(item, (list, tuple)):
                normalized.append(list(item))
            else:
                normalized.append([item])
        return normalized

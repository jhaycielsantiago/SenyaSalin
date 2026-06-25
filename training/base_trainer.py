"""
BaseTrainer — abstract base class for all gesture classifiers.

Every trainer must implement:
- build_model(): Returns an sklearn-compatible estimator
- train(X, y): Returns a TrainingResult
- save(path): Persists the trained model

The sklearn interface (fit/predict/predict_proba) is required so that
the benchmark evaluator can treat all models uniformly. Future contributors
wrapping PyTorch/TF models must implement an sklearn-compatible wrapper.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)


@dataclass
class TrainingResult:
    model_name: str
    train_accuracy: float
    val_accuracy: float
    val_precision: float
    val_recall: float
    val_f1: float
    inference_time_ms_mean: float
    inference_time_ms_p95: float
    class_names: list[str]
    hyperparameters: dict = field(default_factory=dict)
    classification_report_str: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def print_summary(self) -> None:
        print(f"\n{'─'*50}")
        print(f"  Model: {self.model_name}")
        print(f"  Train Accuracy: {self.train_accuracy:.4f}")
        print(f"  Val   Accuracy: {self.val_accuracy:.4f}")
        print(f"  Val   F1 (weighted): {self.val_f1:.4f}")
        print(f"  Inference: mean={self.inference_time_ms_mean:.2f}ms  p95={self.inference_time_ms_p95:.2f}ms")
        print(f"{'─'*50}")


class BaseTrainer(ABC):
    """
    Abstract base class for gesture classifiers.

    Subclasses:
        - Implement build_model() to return an sklearn estimator.
        - Call self.model = self.build_model() in __init__.
        - All other methods are provided by BaseTrainer.

    Args:
        class_names: List of gesture class names, index = encoded label.
        random_seed: For reproducibility.
    """

    def __init__(self, class_names: list[str], random_seed: int = 42) -> None:
        self.class_names = class_names
        self.random_seed = random_seed
        self.model: Optional[BaseEstimator] = None

    @abstractmethod
    def build_model(self) -> BaseEstimator:
        """Return an untrained sklearn-compatible estimator."""
        ...

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> TrainingResult:
        """Train on X_train/y_train, evaluate on X_val/y_val."""
        self.model = self.build_model()
        self.model.fit(X_train, y_train)

        train_preds = self.model.predict(X_train)
        val_preds = self.model.predict(X_val)

        # Measure inference time on val set
        times = []
        for sample in X_val:
            t0 = time.perf_counter()
            self.model.predict(sample.reshape(1, -1))
            times.append((time.perf_counter() - t0) * 1000)

        times_arr = np.array(times)

        report = classification_report(
            y_val, val_preds, target_names=self.class_names, zero_division=0
        )

        return TrainingResult(
            model_name=self.__class__.__name__,
            train_accuracy=accuracy_score(y_train, train_preds),
            val_accuracy=accuracy_score(y_val, val_preds),
            val_precision=precision_score(y_val, val_preds, average="weighted", zero_division=0),
            val_recall=recall_score(y_val, val_preds, average="weighted", zero_division=0),
            val_f1=f1_score(y_val, val_preds, average="weighted", zero_division=0),
            inference_time_ms_mean=float(times_arr.mean()),
            inference_time_ms_p95=float(np.percentile(times_arr, 95)),
            class_names=self.class_names,
            hyperparameters=self.get_hyperparameters(),
            classification_report_str=report,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._require_trained()
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._require_trained()
        if not hasattr(self.model, "predict_proba"):
            raise NotImplementedError(f"{self.model.__class__.__name__} does not support predict_proba")
        return self.model.predict_proba(X)

    def save(self, path: Path) -> None:
        """Save trained model + metadata to disk."""
        self._require_trained()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)

        meta = {
            "model_name": self.__class__.__name__,
            "class_names": self.class_names,
            "hyperparameters": self.get_hyperparameters(),
        }
        meta_path = path.with_suffix(".meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    def load(self, path: Path) -> None:
        """Load a previously saved model from disk."""
        path = Path(path)
        self.model = joblib.load(path)

        meta_path = path.with_suffix(".meta.json")
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            self.class_names = meta.get("class_names", self.class_names)

    def get_hyperparameters(self) -> dict:
        """Return a dict of current hyperparameters for logging."""
        return {}

    def _require_trained(self) -> None:
        if self.model is None:
            raise RuntimeError("Model has not been trained yet. Call train() first.")

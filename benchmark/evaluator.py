"""
Benchmark evaluator.

Tests a trained model against the held-out test split and computes
the standard SenyaSalin benchmark metrics defined in schemas/benchmark_config.json.

This module is intentionally separate from training so that benchmark
evaluation can be run independently — e.g., on a model trained elsewhere
or contributed by the community.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from training.base_trainer import BaseTrainer, TrainingResult


class BenchmarkEvaluator:
    """
    Evaluates a trained BaseTrainer against a test split.

    Args:
        trainer: A trained BaseTrainer (or subclass).
    """

    def __init__(self, trainer: BaseTrainer) -> None:
        self.trainer = trainer

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> dict:
        """
        Run full benchmark evaluation on the test split.

        Returns a dict with all standard benchmark metrics.
        """
        preds = self.trainer.predict(X_test)

        # Measure per-sample inference time
        times = []
        for sample in X_test:
            t0 = time.perf_counter()
            self.trainer.predict(sample.reshape(1, -1))
            times.append((time.perf_counter() - t0) * 1000)

        times_arr = np.array(times)
        class_names = self.trainer.class_names

        report = classification_report(
            y_test, preds, target_names=class_names, zero_division=0, output_dict=True
        )
        report_str = classification_report(
            y_test, preds, target_names=class_names, zero_division=0
        )
        cm = confusion_matrix(y_test, preds).tolist()

        return {
            "model_name": self.trainer.__class__.__name__,
            "num_test_samples": len(X_test),
            "class_names": class_names,
            "metrics": {
                "accuracy": accuracy_score(y_test, preds),
                "precision_weighted": precision_score(y_test, preds, average="weighted", zero_division=0),
                "recall_weighted": recall_score(y_test, preds, average="weighted", zero_division=0),
                "f1_weighted": f1_score(y_test, preds, average="weighted", zero_division=0),
                "inference_time_ms_mean": float(times_arr.mean()),
                "inference_time_ms_p95": float(np.percentile(times_arr, 95)),
            },
            "per_class_metrics": {
                cls: {
                    "precision": report[cls]["precision"],
                    "recall": report[cls]["recall"],
                    "f1": report[cls]["f1-score"],
                    "support": report[cls]["support"],
                }
                for cls in class_names
                if cls in report
            },
            "confusion_matrix": cm,
            "classification_report": report_str,
        }

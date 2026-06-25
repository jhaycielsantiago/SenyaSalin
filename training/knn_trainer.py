"""
KNN baseline trainer.

K-Nearest Neighbors is the simplest baseline for gesture recognition.
It stores all training feature vectors and classifies new inputs by
voting among the K nearest neighbors (Euclidean distance by default).

Why KNN first: it requires no training time, is easily interpretable,
and acts as a strong sanity check — if KNN can't classify the gestures,
the features or normalization need improvement before trying complex models.
"""

from __future__ import annotations

from sklearn.neighbors import KNeighborsClassifier

from .base_trainer import BaseTrainer


class KNNTrainer(BaseTrainer):
    """
    K-Nearest Neighbors gesture classifier.

    Args:
        k: Number of neighbors to consider.
        metric: Distance metric ("euclidean", "cosine", "manhattan").
        weights: How to weight neighbors ("uniform" or "distance").
        class_names: Gesture class names.
        random_seed: Unused for KNN, kept for consistent interface.
    """

    def __init__(
        self,
        class_names: list[str],
        k: int = 5,
        metric: str = "euclidean",
        weights: str = "distance",
        random_seed: int = 42,
    ) -> None:
        super().__init__(class_names, random_seed)
        self.k = k
        self.metric = metric
        self.weights = weights

    def build_model(self) -> KNeighborsClassifier:
        return KNeighborsClassifier(
            n_neighbors=self.k,
            metric=self.metric,
            weights=self.weights,
            algorithm="ball_tree",
            n_jobs=-1,
        )

    def get_hyperparameters(self) -> dict:
        return {"k": self.k, "metric": self.metric, "weights": self.weights}

"""
Random Forest baseline trainer.

Random Forest provides a strong, interpretable baseline with feature importance
scores — useful for understanding which hand landmarks and joints are most
discriminative for each gesture class.

Future contributors: Use feature importances from the trained RF model to guide
feature selection for lightweight deployment models.
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier

from .base_trainer import BaseTrainer


class RandomForestTrainer(BaseTrainer):
    """
    Random Forest gesture classifier.

    Args:
        n_estimators: Number of decision trees.
        max_depth: Max tree depth (None = unlimited).
        min_samples_split: Min samples to split an internal node.
        class_names: Gesture class names.
        random_seed: For reproducibility.
    """

    def __init__(
        self,
        class_names: list[str],
        n_estimators: int = 200,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        random_seed: int = 42,
    ) -> None:
        super().__init__(class_names, random_seed)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def build_model(self) -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            random_state=self.random_seed,
            n_jobs=-1,
            class_weight="balanced",
        )

    def get_hyperparameters(self) -> dict:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
        }

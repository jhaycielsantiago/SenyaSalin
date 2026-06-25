"""
MLP baseline trainer.

Multi-Layer Perceptron provides a non-linear baseline that can capture
feature interactions the linear KNN and RF baselines may miss.
Uses sklearn's MLPClassifier (CPU-based, no GPU required) so it runs
on any researcher's laptop without deep learning setup.

Future contributors: Replace this with a PyTorch or TF-based MLP for GPU
acceleration and more flexible architectures. The BaseTrainer interface
remains the same — just swap the sklearn estimator for an sklearn-wrapped
PyTorch model.
"""

from __future__ import annotations

from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .base_trainer import BaseTrainer


class MLPTrainer(BaseTrainer):
    """
    Multi-Layer Perceptron gesture classifier.

    Includes a StandardScaler in a Pipeline because MLPs are sensitive
    to input scale — the scaler is saved with the model automatically.

    Args:
        hidden_layer_sizes: Tuple defining layer widths e.g. (256, 128).
        max_iter: Max training epochs.
        learning_rate_init: Initial learning rate for Adam.
        class_names: Gesture class names.
        random_seed: For reproducibility.
    """

    def __init__(
        self,
        class_names: list[str],
        hidden_layer_sizes: tuple[int, ...] = (256, 128),
        max_iter: int = 500,
        learning_rate_init: float = 0.001,
        random_seed: int = 42,
    ) -> None:
        super().__init__(class_names, random_seed)
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.learning_rate_init = learning_rate_init

    def build_model(self) -> Pipeline:
        mlp = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation="relu",
            solver="adam",
            max_iter=self.max_iter,
            learning_rate_init=self.learning_rate_init,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=self.random_seed,
        )
        return Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", mlp),
        ])

    def get_hyperparameters(self) -> dict:
        return {
            "hidden_layer_sizes": list(self.hidden_layer_sizes),
            "max_iter": self.max_iter,
            "learning_rate_init": self.learning_rate_init,
        }

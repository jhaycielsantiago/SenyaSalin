"""
Training module.

Provides base trainer infrastructure and three baseline model trainers:
- KNNTrainer
- RandomForestTrainer
- MLPTrainer

Future contributors: subclass BaseTrainer to add new model types.
All trainers save models to models/saved/ in a consistent format.
"""

from .base_trainer import BaseTrainer, TrainingResult
from .knn_trainer import KNNTrainer
from .rf_trainer import RandomForestTrainer
from .mlp_trainer import MLPTrainer

__all__ = [
    "BaseTrainer",
    "TrainingResult",
    "KNNTrainer",
    "RandomForestTrainer",
    "MLPTrainer",
]

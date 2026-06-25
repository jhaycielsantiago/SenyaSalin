"""
Dataset loader.

Loads processed landmark JSON files from data/processed/ and prepares
(X, y) arrays for training and evaluation.

Supports:
- Stratified train/val/test splitting (reproducible via random seed)
- Label encoding
- Per-split CSV export for benchmark reproducibility
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .feature_extractor import FeatureExtractor, AggregationStrategy
from extraction.landmark_extractor import GestureLandmarkRecord


class DataLoader:
    """
    Loads a processed landmark dataset into numpy arrays.

    Args:
        processed_dir: Path to data/processed/
        feature_extractor: FeatureExtractor to use (defaults to mean aggregation).
        random_seed: For reproducible splits.
    """

    def __init__(
        self,
        processed_dir: str | Path = "data/processed",
        feature_extractor: Optional[FeatureExtractor] = None,
        random_seed: int = 42,
    ) -> None:
        self.processed_dir = Path(processed_dir)
        self.feature_extractor = feature_extractor or FeatureExtractor()
        self.random_seed = random_seed
        self.label_encoder = LabelEncoder()

    def load_all(self) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """
        Load all landmark records from processed_dir.

        Returns:
            X: Feature matrix of shape (N, feature_dim).
            y: Integer label array of shape (N,).
            class_names: List of gesture labels (index = encoded class).
        """
        X_list: list[np.ndarray] = []
        y_list: list[str] = []

        label_dirs = sorted([d for d in self.processed_dir.iterdir() if d.is_dir()])
        if not label_dirs:
            raise RuntimeError(
                f"No label directories found in {self.processed_dir}. "
                "Run extraction first."
            )

        for label_dir in label_dirs:
            label = label_dir.name
            json_files = sorted(label_dir.glob("*.json"))
            for json_file in json_files:
                try:
                    _, features = self.feature_extractor.extract_from_file(json_file)
                    X_list.append(features)
                    y_list.append(label)
                except Exception as e:
                    print(f"[WARN] Skipping {json_file}: {e}")

        if not X_list:
            raise RuntimeError("No data loaded. Check data/processed/ directory.")

        X = np.stack(X_list, axis=0)
        y_encoded = self.label_encoder.fit_transform(y_list)
        class_names = list(self.label_encoder.classes_)

        return X, y_encoded, class_names

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """
        Stratified train/val/test split.

        Returns:
            {"train": (X_train, y_train), "val": (X_val, y_val), "test": (X_test, y_test)}
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            test_size=(1.0 - train_ratio),
            stratify=y,
            random_state=self.random_seed,
        )

        relative_val = val_ratio / (val_ratio + test_ratio)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=(1.0 - relative_val),
            stratify=y_temp,
            random_state=self.random_seed,
        )

        return {
            "train": (X_train, y_train),
            "val": (X_val, y_val),
            "test": (X_test, y_test),
        }

    def export_splits_csv(
        self,
        splits: dict[str, tuple[np.ndarray, np.ndarray]],
        class_names: list[str],
        output_dir: Path,
    ) -> None:
        """
        Export each split to a CSV for benchmark reproducibility.

        Columns: [feature_0, ..., feature_N, label_encoded, label_name]
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for split_name, (X, y) in splits.items():
            df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
            df["label_encoded"] = y
            df["label_name"] = [class_names[i] for i in y]
            out_path = output_dir / f"{split_name}.csv"
            df.to_csv(out_path, index=False)
            print(f"Exported {split_name}: {len(df)} samples → {out_path}")

"""
Training CLI entry point.

Trains all baseline models on the processed dataset and saves them to models/saved/.
Optionally exports benchmark-compatible splits and result JSON.

Usage:
    python -m training.train
    python -m training.train --model knn
    python -m training.train --processed-dir data/processed --output-dir models/saved
    python -m training.train --export-splits
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from preprocessing.data_loader import DataLoader
from preprocessing.feature_extractor import FeatureExtractor
from preprocessing.normalizer import LandmarkNormalizer
from training import KNNTrainer, RandomForestTrainer, MLPTrainer, TrainingResult


TRAINERS = {
    "knn": KNNTrainer,
    "random_forest": RandomForestTrainer,
    "mlp": MLPTrainer,
}


def run_training(
    processed_dir: Path,
    output_dir: Path,
    model_names: list[str],
    export_splits: bool = False,
    benchmark_dir: Path = Path("data/benchmark"),
    random_seed: int = 42,
) -> dict[str, TrainingResult]:
    """
    Load data, train models, save results.

    Returns a dict of {model_name: TrainingResult}.
    """
    print(f"\nLoading dataset from: {processed_dir}")
    normalizer = LandmarkNormalizer(strategy="wrist_relative_unit_scale")
    extractor = FeatureExtractor(normalizer=normalizer, aggregation="mean")
    loader = DataLoader(
        processed_dir=processed_dir,
        feature_extractor=extractor,
        random_seed=random_seed,
    )

    X, y, class_names = loader.load_all()
    print(f"Loaded {len(X)} samples across {len(class_names)} classes: {class_names}")

    splits = loader.split(X, y)
    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    X_test, y_test = splits["test"]
    print(f"Split: train={len(X_train)}  val={len(X_val)}  test={len(X_test)}")

    if export_splits:
        loader.export_splits_csv(splits, class_names, benchmark_dir / "splits")

    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, TrainingResult] = {}

    for model_name in model_names:
        print(f"\nTraining: {model_name.upper()}")
        TrainerClass = TRAINERS[model_name]
        trainer = TrainerClass(class_names=class_names, random_seed=random_seed)

        result = trainer.train(X_train, y_train, X_val, y_val)
        result.print_summary()
        results[model_name] = result

        model_path = output_dir / f"{model_name}.pkl"
        trainer.save(model_path)
        print(f"  Saved: {model_path}")

    # Persist training results as JSON for benchmark tracking
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = output_dir / f"training_results_{run_id}.json"
    serialized = {k: v.to_dict() for k, v in results.items()}
    with open(results_path, "w") as f:
        json.dump(serialized, f, indent=2)
    print(f"\nResults saved: {results_path}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SenyaSalin baseline gesture classifiers.")
    parser.add_argument("--processed-dir", default="data/processed", help="Processed landmark dir")
    parser.add_argument("--output-dir", default="models/saved", help="Where to save models")
    parser.add_argument(
        "--model",
        choices=list(TRAINERS.keys()) + ["all"],
        default="all",
        help="Which model to train",
    )
    parser.add_argument("--export-splits", action="store_true", help="Export train/val/test CSVs to data/benchmark/")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    model_names = list(TRAINERS.keys()) if args.model == "all" else [args.model]

    run_training(
        processed_dir=Path(args.processed_dir),
        output_dir=Path(args.output_dir),
        model_names=model_names,
        export_splits=args.export_splits,
        random_seed=args.seed,
    )


if __name__ == "__main__":
    main()

"""
Benchmark CLI entry point.

Loads trained models from models/saved/ and evaluates them against
the held-out test split. Results are saved to benchmark/results/.

Usage:
    python -m benchmark.run_benchmark
    python -m benchmark.run_benchmark --model knn
    python -m benchmark.run_benchmark --compare benchmark/results/*.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from preprocessing.data_loader import DataLoader
from preprocessing.feature_extractor import FeatureExtractor
from preprocessing.normalizer import LandmarkNormalizer
from training import KNNTrainer, RandomForestTrainer, MLPTrainer
from training.base_trainer import BaseTrainer
from benchmark.evaluator import BenchmarkEvaluator
from benchmark.metrics import MetricsReporter


TRAINER_CLASSES = {
    "knn": KNNTrainer,
    "random_forest": RandomForestTrainer,
    "mlp": MLPTrainer,
}


def run_benchmark(
    processed_dir: Path,
    models_dir: Path,
    results_dir: Path,
    model_names: list[str],
    random_seed: int = 42,
) -> None:
    normalizer = LandmarkNormalizer(strategy="wrist_relative_unit_scale")
    extractor = FeatureExtractor(normalizer=normalizer, aggregation="mean")
    loader = DataLoader(processed_dir=processed_dir, feature_extractor=extractor, random_seed=random_seed)

    X, y, class_names = loader.load_all()
    splits = loader.split(X, y)
    _, (X_test, y_test) = list(splits.items())[2]  # test split

    print(f"Benchmark test split: {len(X_test)} samples, {len(class_names)} classes")

    for model_name in model_names:
        model_path = models_dir / f"{model_name}.pkl"
        if not model_path.exists():
            print(f"[SKIP] {model_name}: no saved model at {model_path}. Train first.")
            continue

        TrainerClass = TRAINER_CLASSES[model_name]
        trainer = TrainerClass(class_names=class_names)
        trainer.load(model_path)

        evaluator = BenchmarkEvaluator(trainer)
        results = evaluator.evaluate(X_test, y_test)

        MetricsReporter.print_results(results)
        MetricsReporter.save_results(results, results_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SenyaSalin benchmark evaluation.")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--models-dir", default="models/saved")
    parser.add_argument("--results-dir", default="benchmark/results")
    parser.add_argument("--model", choices=list(TRAINER_CLASSES.keys()) + ["all"], default="all")
    parser.add_argument(
        "--compare",
        nargs="+",
        metavar="RESULT_FILE",
        help="Compare multiple benchmark result JSON files",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.compare:
        MetricsReporter.compare_results([Path(p) for p in args.compare])
        return

    model_names = list(TRAINER_CLASSES.keys()) if args.model == "all" else [args.model]
    run_benchmark(
        processed_dir=Path(args.processed_dir),
        models_dir=Path(args.models_dir),
        results_dir=Path(args.results_dir),
        model_names=model_names,
        random_seed=args.seed,
    )


if __name__ == "__main__":
    main()

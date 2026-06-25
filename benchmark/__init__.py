"""
Benchmark module.

Provides the evaluation harness for comparing gesture classifiers
against the SenyaSalin benchmark dataset.

Key components:
- BenchmarkEvaluator: Evaluates any BaseTrainer on the test split
- MetricsReporter: Formats and exports benchmark results
- run_benchmark.py: CLI entry point

All benchmark results are versioned and stored in benchmark/results/.
Future contributors: run the benchmark against new models and commit
the result JSON so reviewers can compare performance over time.
"""

from .evaluator import BenchmarkEvaluator
from .metrics import MetricsReporter

__all__ = ["BenchmarkEvaluator", "MetricsReporter"]

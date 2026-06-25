"""
Metrics formatting and result persistence.

Handles human-readable benchmark output and versioned JSON result storage.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


class MetricsReporter:
    """
    Formats and persists benchmark evaluation results.
    """

    @staticmethod
    def print_results(results: dict) -> None:
        model = results["model_name"]
        m = results["metrics"]
        n = results["num_test_samples"]

        print(f"\n{'='*60}")
        print(f"  BENCHMARK RESULTS: {model}")
        print(f"  Test samples: {n}")
        print(f"{'='*60}")
        print(f"  Accuracy:   {m['accuracy']:.4f}")
        print(f"  Precision:  {m['precision_weighted']:.4f}  (weighted)")
        print(f"  Recall:     {m['recall_weighted']:.4f}  (weighted)")
        print(f"  F1:         {m['f1_weighted']:.4f}  (weighted)")
        print(f"  Inference:  mean={m['inference_time_ms_mean']:.2f}ms  p95={m['inference_time_ms_p95']:.2f}ms")
        print(f"{'='*60}")
        print("\nPer-class results:")
        for cls, cm in results.get("per_class_metrics", {}).items():
            print(f"  {cls:<20} P={cm['precision']:.3f}  R={cm['recall']:.3f}  F1={cm['f1']:.3f}  n={int(cm['support'])}")
        print()

    @staticmethod
    def save_results(results: dict, results_dir: Path, run_tag: str = "") -> Path:
        """
        Save benchmark results to a versioned JSON file.

        Args:
            results: Result dict from BenchmarkEvaluator.evaluate().
            results_dir: Directory to save results in (benchmark/results/).
            run_tag: Optional tag to identify the run (e.g. commit hash or date).

        Returns:
            Path to the saved file.
        """
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_slug = results["model_name"].lower().replace(" ", "_")
        tag = f"_{run_tag}" if run_tag else ""
        filename = f"{model_slug}{tag}_{timestamp}.json"

        out_path = results_dir / filename
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"Benchmark result saved: {out_path}")
        return out_path

    @staticmethod
    def compare_results(result_files: list[Path]) -> None:
        """
        Print a comparison table from multiple benchmark result JSON files.
        """
        rows = []
        for path in result_files:
            with open(path) as f:
                r = json.load(f)
            m = r["metrics"]
            rows.append({
                "model": r["model_name"],
                "accuracy": m["accuracy"],
                "f1": m["f1_weighted"],
                "inf_mean_ms": m["inference_time_ms_mean"],
            })

        print(f"\n{'Model':<25} {'Accuracy':>10} {'F1 (w)':>10} {'Inf (ms)':>10}")
        print("─" * 60)
        for row in sorted(rows, key=lambda r: r["f1"], reverse=True):
            print(
                f"  {row['model']:<23} {row['accuracy']:>10.4f} {row['f1']:>10.4f} {row['inf_mean_ms']:>10.2f}"
            )
        print()

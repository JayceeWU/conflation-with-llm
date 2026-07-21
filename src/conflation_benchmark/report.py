from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import METRIC_NAMES, bootstrap_intervals, classification_metrics
from .runtime import validate_prediction_columns


def read_predictions(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    frame = pd.read_json(path, lines=True) if path.suffix == ".jsonl" else pd.read_csv(path)
    validate_prediction_columns(frame.columns.tolist())
    return frame


def evaluate_file(path: str | Path, bootstrap_iterations: int = 1000) -> dict[str, object]:
    frame = read_predictions(path)
    valid = frame[frame["valid_output"].astype(str).str.lower().isin({"true", "1"})]
    if valid.empty:
        raise ValueError("Prediction file has no valid outputs")
    result: dict[str, object] = classification_metrics(valid.label, valid.prediction)
    result["invalid_output_rate"] = float(1 - len(valid) / len(frame))
    result["bootstrap_95"] = bootstrap_intervals(valid.label, valid.prediction, bootstrap_iterations)
    result["latency_p50_ms"] = float(frame.latency_ms.median())
    result["latency_p95_ms"] = float(frame.latency_ms.quantile(0.95))
    result["throughput_per_second"] = float(1000 / frame.latency_ms.mean())
    result["average_input_tokens"] = float(frame.input_tokens.mean())
    result["average_output_tokens"] = float(frame.output_tokens.mean())
    return result


def _aggregate(prediction_files: list[Path], repository_root: Path | None = None) -> pd.DataFrame:
    rows = []
    for path in prediction_files:
        frame = read_predictions(path)
        valid = frame[frame.valid_output.astype(str).str.lower().isin({"true", "1"})]
        if valid.empty:
            continue
        metrics = classification_metrics(valid.label, valid.prediction)
        intervals = bootstrap_intervals(valid.label, valid.prediction)
        first = frame.iloc[0]
        rows.append({
            "model_id": first.model_id, "track": first.track, "regime": first.regime,
            "scenario": first.scenario, "seed": int(first.seed), **metrics,
            "invalid_output_rate": 1 - len(valid) / len(frame),
            "latency_p50_ms": frame.latency_ms.median(), "latency_p95_ms": frame.latency_ms.quantile(.95),
            "throughput_per_second": 1000 / frame.latency_ms.mean(),
            "average_input_tokens": frame.input_tokens.mean(), "average_output_tokens": frame.output_tokens.mean(),
            "prediction_file": str(path.relative_to(repository_root)) if repository_root else str(path),
        })
        for metric, bounds in intervals.items():
            rows[-1][f"{metric}_ci95_low"] = bounds["low"]
            rows[-1][f"{metric}_ci95_high"] = bounds["high"]
    return pd.DataFrame(rows)


def generate_report(output_dir: str | Path) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    files = sorted(output_dir.glob("**/predictions.csv"))
    if not files:
        files = sorted(output_dir.glob("**/predictions.jsonl"))
    if not files:
        raise FileNotFoundError(f"No prediction files found under {output_dir}")
    runs = _aggregate(files, output_dir.parent)
    report_dir = output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(report_dir / "run_metrics.csv", index=False)
    keys = ["model_id", "track", "regime", "scenario"]
    numeric = list(METRIC_NAMES) + ["invalid_output_rate", "latency_p50_ms", "latency_p95_ms", "throughput_per_second"]
    summary = runs.groupby(keys, as_index=False)[numeric].agg(["mean", "std"])
    summary.columns = ["_".join(filter(None, map(str, col))).rstrip("_") for col in summary.columns]
    summary.to_csv(report_dir / "summary.csv", index=False)
    full = runs[runs.scenario == "full"].sort_values("f1", ascending=False)
    lines = ["# Reproducible Small-Model Benchmark", "", "Generated only from prediction artifacts.", "", "## Full-field runs", ""]
    columns = ["model_id", "track", "regime", "seed", "accuracy", "precision", "recall", "f1", "f1_ci95_low", "f1_ci95_high", "balanced_accuracy", "mcc", "invalid_output_rate", "latency_p50_ms"]
    lines.append(full[columns].to_markdown(index=False, floatfmt=".4f") if len(full) else "No full-field runs found.")
    lines.extend(["", "## Traceability", "", "Machine-readable source: `run_metrics.csv`; aggregate: `summary.csv`.", ""])
    report_path = report_dir / "benchmark.md"
    report_path.write_text("\n".join(lines))
    _plot_metrics(runs, report_dir)
    return report_path, report_dir / "summary.csv"


def _plot_metrics(runs: pd.DataFrame, output: Path) -> None:
    import matplotlib.pyplot as plt
    for metric in ("accuracy", "precision", "recall", "f1"):
        pivot = runs.groupby(["model_id", "scenario"])[metric].mean().unstack(fill_value=np.nan)
        ax = pivot.plot(kind="bar", figsize=(12, 6))
        ax.set_ylabel(metric)
        ax.set_ylim(0, 1)
        ax.set_title(f"{metric.title()} by model and scenario")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(output / f"{metric}.png", dpi=200)
        plt.close()

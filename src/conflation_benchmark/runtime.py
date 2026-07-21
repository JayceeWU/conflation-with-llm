from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any


PREDICTION_COLUMNS = [
    "example_id", "split", "model_id", "track", "regime", "scenario", "seed",
    "label", "prediction", "valid_output", "score_or_raw_output", "input_tokens",
    "output_tokens", "latency_ms",
]


def safe_model_name(model_id: str) -> str:
    return model_id.replace("/", "__")


def cached_model_revision(model_id: str) -> str | None:
    """Return the commit pinned by the local Hugging Face `main` cache ref."""
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    ref = hf_home / "hub" / f"models--{model_id.replace('/', '--')}" / "refs" / "main"
    try:
        return ref.read_text().strip() or None
    except OSError:
        return None


def git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment_metadata(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    packages = {}
    distributions = {
        "torch": "torch", "transformers": "transformers", "pandas": "pandas",
        "sklearn": "scikit-learn", "datasets": "datasets", "pyarrow": "pyarrow",
    }
    for name, distribution in distributions.items():
        try:
            packages[name] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    result: dict[str, Any] = {
        "python": platform.python_version(), "platform": platform.platform(),
        "git_revision": git_revision(), "packages": packages,
    }
    try:
        import torch
        result["cuda_available"] = torch.cuda.is_available()
        result["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        result["cuda_available"] = False
        result["gpu"] = None
    if extra:
        result.update(extra)
    return result


def write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True, default=str) + "\n")


def validate_prediction_columns(columns: list[str]) -> None:
    missing = set(PREDICTION_COLUMNS) - set(columns)
    if missing:
        raise ValueError(f"Prediction file is missing columns: {sorted(missing)}")

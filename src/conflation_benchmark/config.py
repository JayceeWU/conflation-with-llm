from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

FIELDS = ("name", "category", "website", "social", "email", "phone", "brand", "address")
COLUMNS = {
    "name": ("names", "base_names"),
    "category": ("categories", "base_categories"),
    "website": ("websites", "base_websites"),
    "social": ("socials", "base_socials"),
    "email": ("emails", "base_emails"),
    "phone": ("phones", "base_phones"),
    "brand": ("brand", "base_brand"),
    "address": ("addresses", "base_addresses"),
}
SCENARIOS = {"full": None, **{f"no_{field}": field for field in ("email", "category", "website", "address", "brand")}}


@dataclass(frozen=True)
class Config:
    data_path: Path
    manifest_path: Path
    output_dir: Path
    seed: int
    split_ratios: dict[str, float]
    max_length: int
    encoder_models: tuple[str, ...]
    prompt_models: tuple[str, ...]
    scenarios: tuple[str, ...]


def load_config(path: str | Path = "configs/benchmark.json") -> Config:
    path = Path(path)
    raw = json.loads(path.read_text())
    root = path.parent.parent.resolve()
    def resolve(p: str) -> Path:
        return (root / p).resolve() if not Path(p).is_absolute() else Path(p)
    return Config(
        data_path=resolve(raw["data_path"]),
        manifest_path=resolve(raw["manifest_path"]),
        output_dir=resolve(raw["output_dir"]),
        seed=int(raw["seed"]),
        split_ratios={k: float(v) for k, v in raw["split_ratios"].items()},
        max_length=int(raw["max_length"]),
        encoder_models=tuple(raw["encoder_models"]),
        prompt_models=tuple(raw["prompt_models"]),
        scenarios=tuple(raw["scenarios"]),
    )


def validate_choice(value: str, allowed: tuple[str, ...] | dict[str, object], name: str) -> None:
    if value not in allowed:
        raise ValueError(f"Unknown {name} {value!r}; choose one of: {', '.join(allowed)}")

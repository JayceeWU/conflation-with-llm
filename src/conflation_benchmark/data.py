from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
import pandas as pd

REQUIRED = {"label", "id", "base_id"}


class DisjointSet:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def example_id(row: pd.Series, original_index: int) -> str:
    value = f"{row['id']}\0{row['base_id']}\0{original_index}".encode()
    return hashlib.sha256(value).hexdigest()[:20]


def load_and_clean(path: str | Path) -> tuple[pd.DataFrame, dict[str, object]]:
    frame = pd.read_parquet(path).reset_index(names="source_index")
    raw_rows = len(frame)
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    frame["label"] = pd.to_numeric(frame["label"], errors="coerce")
    invalid_labels = int((~frame["label"].isin([0, 1])).sum())
    missing_ids = int((frame["id"].isna() | frame["base_id"].isna()).sum())
    frame = frame[frame["label"].isin([0, 1]) & frame["id"].notna() & frame["base_id"].notna()].copy()
    frame["label"] = frame["label"].astype(int)
    frame["pair_key"] = frame.apply(lambda r: "\0".join(sorted((str(r["id"]), str(r["base_id"])))), axis=1)
    label_counts = frame.groupby("pair_key")["label"].nunique()
    conflicts = set(label_counts[label_counts > 1].index)
    conflict_rows = int(frame["pair_key"].isin(conflicts).sum())
    frame = frame[~frame["pair_key"].isin(conflicts)].copy()
    before = len(frame)
    frame = frame.drop_duplicates(["pair_key", "label"], keep="first").copy()
    duplicate_rows = before - len(frame)
    frame["example_id"] = [example_id(row, int(row.source_index)) for _, row in frame.iterrows()]
    audit = {
        "raw_rows": int(raw_rows),
        "clean_rows": int(len(frame)),
        "invalid_label_rows": invalid_labels,
        "missing_id_rows": missing_ids,
        "conflicting_pair_rows_removed": conflict_rows,
        "duplicate_pair_rows_removed": duplicate_rows,
        "label_counts": {str(k): int(v) for k, v in frame["label"].value_counts().sort_index().items()},
    }
    return frame.reset_index(drop=True), audit


def _components(frame: pd.DataFrame) -> dict[str, list[int]]:
    dsu = DisjointSet()
    for row in frame.itertuples():
        dsu.union(str(row.id), str(row.base_id))
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in frame.iterrows():
        groups[dsu.find(str(row["id"]))].append(index)
    return groups


def create_manifest(frame: pd.DataFrame, ratios: dict[str, float], seed: int) -> pd.DataFrame:
    if set(ratios) != {"train", "validation", "test"} or abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise ValueError("split ratios must contain train/validation/test and sum to 1")
    groups = _components(frame)
    rng = random.Random(seed)
    items = list(groups.items())
    rng.shuffle(items)
    items.sort(key=lambda item: len(item[1]), reverse=True)
    targets = {name: len(frame) * ratio for name, ratio in ratios.items()}
    positive_target = {name: frame["label"].sum() * ratio for name, ratio in ratios.items()}
    counts = Counter()
    positives = Counter()
    assignments: dict[int, str] = {}
    for component, indices in items:
        component_positive = int(frame.loc[indices, "label"].sum())
        def score(split: str) -> float:
            # Score the complete allocation, not only the candidate bucket. Scoring
            # one bucket in isolation overfills the smallest split before moving on.
            size_cost = sum(
                ((counts[name] + (len(indices) if name == split else 0) - targets[name]) / max(targets[name], 1)) ** 2
                for name in ratios
            )
            pos_cost = sum(
                ((positives[name] + (component_positive if name == split else 0) - positive_target[name]) / max(positive_target[name], 1)) ** 2
                for name in ratios
            )
            return size_cost + pos_cost
        chosen = min(ratios, key=lambda split: (score(split), counts[split], split))
        counts[chosen] += len(indices)
        positives[chosen] += component_positive
        assignments.update({index: chosen for index in indices})
    manifest = frame[["example_id", "source_index", "id", "base_id", "label"]].copy()
    manifest["split"] = [assignments[i] for i in frame.index]
    validate_manifest(manifest)
    actual = manifest["split"].value_counts(normalize=True)
    tolerance = max(len(indices) for indices in groups.values()) / max(len(frame), 1) + 0.01
    if any(abs(float(actual.get(name, 0)) - ratio) > tolerance for name, ratio in ratios.items()):
        raise ValueError(f"Unable to satisfy split ratios without entity leakage; actual={actual.to_dict()}")
    return manifest.sort_values("example_id").reset_index(drop=True)


def validate_manifest(manifest: pd.DataFrame) -> None:
    entity_splits: dict[str, set[str]] = defaultdict(set)
    for row in manifest.itertuples():
        entity_splits[str(row.id)].add(row.split)
        entity_splits[str(row.base_id)].add(row.split)
    leaked = [entity for entity, splits in entity_splits.items() if len(splits) > 1]
    if leaked:
        raise ValueError(f"Entity leakage across splits ({len(leaked)} entities); first: {leaked[:3]}")
    if manifest["example_id"].duplicated().any():
        raise ValueError("Duplicate example_id values in manifest")


def attach_manifest(frame: pd.DataFrame, manifest_path: str | Path) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path, dtype={"id": str, "base_id": str})
    validate_manifest(manifest)
    merged = frame.merge(manifest[["example_id", "split"]], on="example_id", how="inner", validate="one_to_one")
    if len(merged) != len(frame):
        raise ValueError("Manifest does not exactly match cleaned dataset; rerun validate-data")
    return merged


def write_audit(audit: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")

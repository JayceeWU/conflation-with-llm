from __future__ import annotations

import json
import math
from typing import Any, Mapping

from .config import COLUMNS, FIELDS, SCENARIOS, validate_choice


def _stable_value(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "<EMPTY>"
    text = str(value).strip()
    if not text:
        return "<EMPTY>"
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return text
    if parsed in (None, "", [], {}):
        return "<EMPTY>"
    return json.dumps(parsed, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def serialize_record(row: Mapping[str, Any], side: str, scenario: str = "full") -> str:
    validate_choice(scenario, SCENARIOS, "scenario")
    if side not in {"a", "b"}:
        raise ValueError("side must be 'a' or 'b'")
    excluded = SCENARIOS[scenario]
    lines = [f"[RECORD_{side.upper()}]"]
    for field in FIELDS:
        if field == excluded:
            continue
        column = COLUMNS[field][0 if side == "a" else 1]
        lines.append(f"{field.upper()}: {_stable_value(row.get(column))}")
    return "\n".join(lines)


def serialize_pair(row: Mapping[str, Any], scenario: str = "full") -> str:
    return f"{serialize_record(row, 'a', scenario)}\n[PAIR_SEPARATOR]\n{serialize_record(row, 'b', scenario)}"

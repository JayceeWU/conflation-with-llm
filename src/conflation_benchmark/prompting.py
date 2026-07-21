from __future__ import annotations

import difflib
import time
from pathlib import Path
from typing import Any

import pandas as pd

from .config import Config, SCENARIOS, validate_choice
from .data import attach_manifest, load_and_clean
from .metrics import classification_metrics
from .runtime import cached_model_revision, environment_metadata, safe_model_name, write_metadata
from .serialization import serialize_pair

SYSTEM_PROMPT = (
    "You decide whether two place records refer to the same real-world place. "
    "Return exactly MATCH when they are the same place, otherwise return exactly NO_MATCH. "
    "Do not return punctuation, JSON, or an explanation."
)


def parse_label(text: str) -> tuple[int, bool]:
    normalized = text.strip()
    if normalized == "MATCH":
        return 1, True
    if normalized == "NO_MATCH":
        return 0, True
    return 0, False


def _name_similarity(row: pd.Series) -> float:
    return difflib.SequenceMatcher(None, str(row.get("names", "")), str(row.get("base_names", ""))).ratio()


def select_three_shot(train: pd.DataFrame) -> list[pd.Series]:
    positives = train[train.label == 1]
    negatives = train[train.label == 0].copy()
    if positives.empty or len(negatives) < 2:
        raise ValueError("3-shot prompting requires at least one positive and two negative training examples")
    negatives["_similarity"] = negatives.apply(_name_similarity, axis=1)
    positive = positives.sort_values("example_id").iloc[0]
    obvious_negative = negatives.sort_values(["_similarity", "example_id"], ascending=[True, True]).iloc[0]
    hard_negative = negatives.sort_values(["_similarity", "example_id"], ascending=[False, True]).iloc[0]
    return [positive, obvious_negative, hard_negative]


def build_messages(row: pd.Series, scenario: str, examples: list[pd.Series] | None = None) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for example in examples or []:
        messages.append({"role": "user", "content": serialize_pair(example, scenario)})
        messages.append({"role": "assistant", "content": "MATCH" if int(example.label) == 1 else "NO_MATCH"})
    messages.append({"role": "user", "content": serialize_pair(row, scenario)})
    return messages


def _load_model(model_id: str, torch):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto",
        )
    except Exception as exc:
        if model_id.startswith(("meta-llama/", "google/gemma")):
            raise RuntimeError(
                f"Unable to load gated model {model_id}. Accept its Hugging Face license and authenticate with `hf auth login`."
            ) from exc
        raise
    if tokenizer.chat_template is None:
        raise RuntimeError(f"{model_id} does not provide an official chat template")
    return tokenizer, model


def run_prompt(config: Config, model_id: str, regime: str, scenario: str) -> Path:
    validate_choice(model_id, config.prompt_models, "prompt model")
    validate_choice(scenario, SCENARIOS, "scenario")
    if regime not in {"zero", "three-shot"}:
        raise ValueError("regime must be zero or three-shot")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Prompt-track BF16 evaluation requires a CUDA GPU")
    frame, _ = load_and_clean(config.data_path)
    frame = attach_manifest(frame, config.manifest_path)
    train = frame[frame.split == "train"].copy()
    test = frame[frame.split == "test"].copy().sort_values("example_id")
    examples = select_three_shot(train) if regime == "three-shot" else []
    tokenizer, model = _load_model(model_id, torch)
    device = next(model.parameters()).device

    # Use a real test-shaped prompt for warm-up, without reading its label.
    warm_messages = build_messages(test.iloc[0], scenario, examples)
    warm_inputs = tokenizer.apply_chat_template(
        warm_messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(device)
    with torch.inference_mode():
        for _ in range(20):
            model.generate(**warm_inputs, max_new_tokens=4, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)

    records: list[dict[str, Any]] = []
    for _, row in test.iterrows():
        messages = build_messages(row, scenario, examples)
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **inputs, max_new_tokens=4, do_sample=False, pad_token_id=tokenizer.eos_token_id
            )
        if device.type == "cuda":
            torch.cuda.synchronize()
        latency = (time.perf_counter() - start) * 1000
        prompt_length = inputs["input_ids"].shape[-1]
        output_ids = generated[0, prompt_length:]
        raw = tokenizer.decode(output_ids, skip_special_tokens=True)
        prediction, valid = parse_label(raw)
        records.append({
            "example_id": row.example_id, "split": "test", "model_id": model_id,
            "track": "prompt", "regime": regime, "scenario": scenario, "seed": config.seed,
            "label": int(row.label), "prediction": prediction, "valid_output": valid,
            "score_or_raw_output": raw, "input_tokens": int(prompt_length),
            "output_tokens": int(len(output_ids)), "latency_ms": latency,
        })
    output = config.output_dir / "prompt" / safe_model_name(model_id) / scenario / regime
    output.mkdir(parents=True, exist_ok=True)
    predictions = pd.DataFrame(records)
    predictions.to_csv(output / "predictions.csv", index=False)
    predictions.to_json(output / "predictions.jsonl", orient="records", lines=True, force_ascii=False)
    valid = predictions[predictions.valid_output]
    metrics = classification_metrics(valid.label, valid.prediction) if len(valid) else None
    peak_mb = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0
    write_metadata(output / "metadata.json", environment_metadata({
        "model_id": model_id, "scenario": scenario, "regime": regime,
        "model_revision": cached_model_revision(model_id),
        "invalid_output_rate": float(1 - predictions.valid_output.mean()),
        "metrics_on_valid_outputs": metrics, "peak_gpu_memory_mb": peak_mb,
        "three_shot_example_ids": [row.example_id for row in examples],
    }))
    return output / "predictions.csv"

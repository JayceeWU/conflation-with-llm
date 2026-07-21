from __future__ import annotations

import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config, SCENARIOS, validate_choice
from .data import attach_manifest, load_and_clean
from .metrics import classification_metrics
from .runtime import cached_model_revision, environment_metadata, safe_model_name, write_metadata
from .serialization import serialize_pair


def _compute_metrics(eval_prediction) -> dict[str, float]:
    predictions = np.argmax(eval_prediction.predictions, axis=-1)
    return {k: float(v) for k, v in classification_metrics(eval_prediction.label_ids, predictions).items() if k in {"accuracy", "precision", "recall", "f1", "balanced_accuracy", "mcc"}}


def _bf16_supported(torch) -> bool:
    return bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())


def _latencies(model, tokenizer, texts: list[str], device, max_length: int, warmups: int = 20) -> tuple[list[float], float]:
    import torch
    if not texts:
        return [], 0.0
    model.eval()
    def encoded(text):
        return tokenizer(
            text, max_length=max_length, truncation=True, return_tensors="pt"
        ).to(device)
    warm = encoded(texts[0])
    with torch.inference_mode():
        for _ in range(warmups):
            model(**warm)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
    values = []
    with torch.inference_mode():
        for text in texts:
            inputs = encoded(text)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            model(**inputs)
            if device.type == "cuda":
                torch.cuda.synchronize()
            values.append((time.perf_counter() - start) * 1000)
    peak = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0
    return values, peak


def train_encoder(config: Config, model_id: str, scenario: str, seed: int) -> Path:
    validate_choice(model_id, config.encoder_models, "encoder model")
    validate_choice(scenario, SCENARIOS, "scenario")
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification, AutoTokenizer, EarlyStoppingCallback,
        Trainer, TrainingArguments, set_seed,
    )

    frame, _ = load_and_clean(config.data_path)
    frame = attach_manifest(frame, config.manifest_path)
    frame["text"] = frame.apply(lambda row: serialize_pair(row, scenario), axis=1)
    datasets = {
        split: Dataset.from_pandas(frame[frame.split == split][["example_id", "text", "label"]], preserve_index=False)
        for split in ("train", "validation", "test")
    }
    is_deberta = "deberta" in model_id.lower()
    tokenizer_options = {"use_fast": not is_deberta}
    if is_deberta:
        tokenizer_options["fix_mistral_regex"] = True
    tokenizer = AutoTokenizer.from_pretrained(model_id, **tokenizer_options)
    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=config.max_length)
    tokenized = {name: data.map(tokenize, batched=True) for name, data in datasets.items()}
    output = config.output_dir / "encoder" / safe_model_name(model_id) / scenario / f"seed-{seed}"
    output.mkdir(parents=True, exist_ok=True)
    use_bf16 = _bf16_supported(torch) and not is_deberta
    model_options = {"num_labels": 2}
    if is_deberta:
        # This checkpoint may otherwise be materialized as FP16 by newer
        # Transformers releases. AdamW updates then overflow on the first step.
        model_options["dtype"] = torch.float32
    candidates = []
    for learning_rate in (2e-5, 5e-5):
        set_seed(seed)
        model = AutoModelForSequenceClassification.from_pretrained(model_id, **model_options)
        run_dir = output / f"lr-{learning_rate:g}"
        args = TrainingArguments(
            output_dir=str(run_dir), learning_rate=learning_rate, per_device_train_batch_size=32,
            per_device_eval_batch_size=32, num_train_epochs=10, weight_decay=0.01,
            eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
            metric_for_best_model="f1", greater_is_better=True, save_total_limit=1,
            report_to="none", seed=seed, data_seed=seed, bf16=use_bf16,
        )
        trainer = Trainer(
            model=model, args=args, train_dataset=tokenized["train"], eval_dataset=tokenized["validation"],
            processing_class=tokenizer, compute_metrics=_compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )
        trainer.train()
        validation = trainer.evaluate()
        candidates.append((float(validation["eval_f1"]), learning_rate, trainer.state.best_model_checkpoint, validation))
        del trainer, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    candidates.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    best_f1, best_lr, best_checkpoint, validation = candidates[0]
    reload_options = {"dtype": torch.float32} if is_deberta else {}
    model = AutoModelForSequenceClassification.from_pretrained(best_checkpoint, **reload_options)
    eval_args = TrainingArguments(
        output_dir=str(output / "evaluation"), per_device_eval_batch_size=32,
        report_to="none", bf16=use_bf16,
    )
    trainer = Trainer(model=model, args=eval_args, processing_class=tokenizer, compute_metrics=_compute_metrics)
    prediction = trainer.predict(tokenized["test"])
    predicted = np.argmax(prediction.predictions, axis=-1)
    probabilities = torch.softmax(torch.tensor(prediction.predictions), dim=-1)[:, 1].numpy()
    test_rows = frame[frame.split == "test"].reset_index(drop=True)
    texts = test_rows["text"].tolist()
    latencies, peak_mb = _latencies(
        trainer.model, tokenizer, texts, next(trainer.model.parameters()).device, config.max_length
    )
    input_tokens = [len(tokenizer(text, max_length=config.max_length, truncation=True)["input_ids"]) for text in texts]
    result = pd.DataFrame({
        "example_id": test_rows.example_id, "split": "test", "model_id": model_id,
        "track": "encoder", "regime": "fine-tuned", "scenario": scenario, "seed": seed,
        "label": test_rows.label.astype(int), "prediction": predicted.astype(int), "valid_output": True,
        "score_or_raw_output": probabilities, "input_tokens": input_tokens, "output_tokens": 0,
        "latency_ms": latencies,
    })
    result.to_csv(output / "predictions.csv", index=False)
    result.to_json(output / "predictions.jsonl", orient="records", lines=True, force_ascii=False)
    write_metadata(output / "metadata.json", environment_metadata({
        "model_id": model_id, "scenario": scenario, "seed": seed, "best_learning_rate": best_lr,
        "training_precision": "bf16" if use_bf16 else "fp32",
        "model_revision": cached_model_revision(model_id),
        "validation_f1": best_f1, "validation_metrics": validation, "peak_gpu_memory_mb": peak_mb,
        "test_metrics": classification_metrics(result.label, result.prediction),
    }))
    trainer.save_model(output / "best_model")
    tokenizer.save_pretrained(output / "best_model")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output / "predictions.csv"

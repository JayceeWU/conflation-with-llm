# Conflation With Small Language Models

This repository benchmarks small language models for deciding whether two Overture-style place records refer to the same real-world place.

> **Result integrity notice:** The original notebook results used an ineffective `df.drop(...)` call and reversed false-positive/false-negative bookkeeping. Those historical numbers are not part of the reproducible leaderboard and must not be compared with new runs. ELECTRA is rerun by the framework below as the only legacy anchor. Files under `code/`, `results/context_results.md`, and the original presentation are retained as historical artifacts only.

## Benchmark tracks

Supervised sequence classification:

- `google/electra-small-discriminator`
- `microsoft/MiniLM-L12-H384-uncased`
- `microsoft/deberta-v3-small`

Deterministic prompt inference (zero-shot and fixed three-shot):

- `meta-llama/Llama-3.2-1B-Instruct`
- `Qwen/Qwen2.5-1.5B-Instruct`
- `google/gemma-2-2b-it`

The benchmark uses a locked entity-grouped 70/15/15 split. IDs connected through either side of a pair are kept in the same split to prevent entity leakage. Model inputs contain names, categories, websites, social accounts, emails, phones, brands, and addresses; source IDs and confidence fields are excluded.

## Latest GPU main run

The current reproducible report contains exactly four full-field, seed-42 runs: the three supervised encoders above and zero-shot `Qwen/Qwen2.5-1.5B-Instruct`. Llama 3.2 and Gemma 2 were intentionally skipped because they require gated Hugging Face access; this run does not include three-shot prompting, extra seeds, or ablations.

See [`artifacts/reports/benchmark.md`](artifacts/reports/benchmark.md) for the generated leaderboard and confidence intervals. Its values are generated from `run_metrics.csv` and the four per-example prediction files, rather than maintained manually in this README.

For methodology, credibility fixes, resource measurements, confusion matrices, and illustrated analysis, see the [detailed Chinese Markdown report](docs/four-model-gpu-experiment-report.md) or [PDF edition](docs/four-model-gpu-experiment-report.pdf).
An [English Markdown report](docs/four-model-gpu-experiment-report-en.md) and [English PDF edition](docs/four-model-gpu-experiment-report-en.pdf) are also available.

## Installation

Python 3.10+ and a CUDA GPU are recommended. Full runs target a single GPU with at least 24 GB memory.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

Llama and Gemma are gated. Accept their licenses on Hugging Face and authenticate before running them:

```bash
hf auth login
```

## Usage

Audit the source data and create the locked split first:

```bash
benchmark validate-data
```

Train an encoder. Full-field headline runs use seeds 42, 43, and 44:

```bash
benchmark train-encoder --model google/electra-small-discriminator --scenario full --seed 42
```

Run deterministic prompt evaluation:

```bash
benchmark run-prompt --model Qwen/Qwen2.5-1.5B-Instruct --regime zero --scenario full
benchmark run-prompt --model Qwen/Qwen2.5-1.5B-Instruct --regime three-shot --scenario full
```

Available scenarios are `full`, `no_email`, `no_category`, `no_website`, `no_address`, and `no_brand`. For encoder ablations use seed 42; run all scenarios independently because the encoder must be retrained when its input schema changes.

Evaluate one prediction artifact or regenerate every report:

```bash
benchmark evaluate --predictions artifacts/encoder/google__electra-small-discriminator/full/seed-42/predictions.csv
benchmark report
```

The report is generated exclusively from prediction artifacts. Outputs include run metadata, raw per-example predictions, aggregate CSV files, Markdown tables, and plots. No API-price estimate is produced because these runs measure local inference.

## Metrics and timing

The primary metric is F1. Accuracy, precision, recall, balanced accuracy, MCC, TP/TN/FP/FN, invalid prompt-output rate, and 1,000-sample bootstrap confidence intervals are also reported. Latency uses batch size 1, 20 warm-up requests, CUDA synchronization, and p50/p95 summaries.

Every prediction row includes:

```text
example_id, split, model_id, track, regime, scenario, seed,
label, prediction, valid_output, score_or_raw_output,
input_tokens, output_tokens, latency_ms
```

## Tests

```bash
pytest
```

Tests cover stable serialization, paired-field ablation, entity leakage, deterministic split generation, metric orientation, bootstrap reproducibility, and strict prompt-output parsing.

## Repository layout

```text
configs/                 Benchmark model and path configuration
src/conflation_benchmark Reproducible CLI implementation
tests/                   Unit tests
data/                    Original read-only parquet dataset
artifacts/               Generated manifests, predictions, metadata, and reports
code/ and results/       Historical notebook artifacts; not leaderboard sources
```

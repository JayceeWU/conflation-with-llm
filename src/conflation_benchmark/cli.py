from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="benchmark", description="Place-conflation model benchmark")
    parser.add_argument("--config", default="configs/benchmark.json", help="benchmark JSON configuration")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate-data", help="audit data and create the locked split manifest")
    encoder = sub.add_parser("train-encoder", help="fine-tune and evaluate an encoder")
    encoder.add_argument("--model", required=True)
    encoder.add_argument("--scenario", default="full")
    encoder.add_argument("--seed", type=int, default=42)
    prompt = sub.add_parser("run-prompt", help="run deterministic instruct-model evaluation")
    prompt.add_argument("--model", required=True)
    prompt.add_argument("--regime", choices=["zero", "three-shot"], required=True)
    prompt.add_argument("--scenario", default="full")
    evaluate = sub.add_parser("evaluate", help="evaluate a prediction CSV or JSONL")
    evaluate.add_argument("--predictions", required=True)
    evaluate.add_argument("--output")
    sub.add_parser("report", help="regenerate reports and plots from prediction artifacts")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_config(args.config)
    if args.command == "validate-data":
        from .data import create_manifest, load_and_clean, write_audit
        frame, audit = load_and_clean(config.data_path)
        manifest = create_manifest(frame, config.split_ratios, config.seed)
        config.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(config.manifest_path, index=False)
        audit["splits"] = {
            split: {"rows": int(len(part)), "positives": int(part.label.sum())}
            for split, part in manifest.groupby("split")
        }
        audit["manifest_path"] = str(config.manifest_path)
        write_audit(audit, config.output_dir / "data_audit.json")
        print(json.dumps(audit, indent=2, sort_keys=True))
        return 0
    if not config.manifest_path.exists() and args.command in {"train-encoder", "run-prompt"}:
        raise SystemExit("Split manifest not found. Run `benchmark validate-data` first.")
    if args.command == "train-encoder":
        from .encoder import train_encoder
        print(train_encoder(config, args.model, args.scenario, args.seed))
        return 0
    if args.command == "run-prompt":
        from .prompting import run_prompt
        print(run_prompt(config, args.model, args.regime, args.scenario))
        return 0
    if args.command == "evaluate":
        from .report import evaluate_file
        result = evaluate_file(args.predictions)
        rendered = json.dumps(result, indent=2, sort_keys=True)
        if args.output:
            Path(args.output).write_text(rendered + "\n")
        print(rendered)
        return 0
    if args.command == "report":
        from .report import generate_report
        report, summary = generate_report(config.output_dir)
        print(f"{report}\n{summary}")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

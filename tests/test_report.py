import pandas as pd

from conflation_benchmark import report


def test_report_is_generated_from_prediction_artifact(tmp_path, monkeypatch):
    run = tmp_path / "encoder" / "demo" / "full" / "seed-42"
    run.mkdir(parents=True)
    frame = pd.DataFrame({
        "example_id": ["a", "b", "c", "d"], "split": "test", "model_id": "demo/model",
        "track": "encoder", "regime": "fine-tuned", "scenario": "full", "seed": 42,
        "label": [1, 1, 0, 0], "prediction": [1, 0, 1, 0], "valid_output": True,
        "score_or_raw_output": [.9, .4, .6, .1], "input_tokens": 20, "output_tokens": 0,
        "latency_ms": [2.0, 2.5, 3.0, 3.5],
    })
    frame.to_csv(run / "predictions.csv", index=False)
    monkeypatch.setattr(report, "_plot_metrics", lambda *_: None)
    markdown, summary = report.generate_report(tmp_path)
    assert "demo/model" in markdown.read_text()
    assert "prediction_file" in (tmp_path / "reports" / "run_metrics.csv").read_text()
    assert summary.exists()

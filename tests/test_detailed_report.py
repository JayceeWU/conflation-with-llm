import re
from pathlib import Path

import pandas as pd
from pypdf import PdfReader


def test_detailed_report_is_traceable_and_complete():
    root = Path(__file__).resolve().parents[1]
    report_path = root / "docs/four-model-gpu-experiment-report.md"
    pdf_path = root / "docs/four-model-gpu-experiment-report.pdf"
    metrics = pd.read_csv(root / "artifacts/reports/run_metrics.csv")
    report = report_path.read_text()

    full = metrics[metrics.scenario == "full"]
    assert len(full) == 4
    for row in full.itertuples(index=False):
        predictions = pd.read_csv(root / row.prediction_file)
        assert len(predictions) == 450
        assert f"{row.f1:.4f}" in report
    assert all(name in report for name in ("MiniLM", "ELECTRA", "DeBERTa", "Qwen"))

    image_links = re.findall(r"!\[[^]]*]\(([^)]+)\)", report)
    assert len(image_links) == 7
    assert all((report_path.parent / link).exists() for link in image_links)

    reader = PdfReader(pdf_path)
    assert len(reader.pages) >= 8
    assert pdf_path.stat().st_size > 200_000
    text = "".join((page.extract_text() or "") for page in reader.pages)
    assert "四模型 GPU 地点匹配实验详细报告" in text
    assert "可信度、限制与解释边界" in text
    assert sum(len(page.images) for page in reader.pages) >= 7


def test_english_report_is_traceable_and_complete():
    root = Path(__file__).resolve().parents[1]
    report_path = root / "docs/four-model-gpu-experiment-report-en.md"
    pdf_path = root / "docs/four-model-gpu-experiment-report-en.pdf"
    metrics = pd.read_csv(root / "artifacts/reports/run_metrics.csv")
    report = report_path.read_text()

    for row in metrics[metrics.scenario == "full"].itertuples(index=False):
        assert f"{row.f1:.4f}" in report
    assert all(name in report for name in ("MiniLM", "ELECTRA", "DeBERTa", "Qwen"))
    image_links = re.findall(r"!\[[^]]*]\(([^)]+)\)", report)
    assert len(image_links) == 7
    assert all((report_path.parent / link).exists() for link in image_links)

    reader = PdfReader(pdf_path)
    assert len(reader.pages) >= 8
    text = "".join((page.extract_text() or "") for page in reader.pages)
    assert "Detailed Four-Model GPU Place-Matching Experiment Report" in text
    assert "Credibility, Limitations, and Interpretation Boundaries" in text
    assert sum(len(page.images) for page in reader.pages) >= 7

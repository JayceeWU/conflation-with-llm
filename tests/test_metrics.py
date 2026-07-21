import pytest

from conflation_benchmark.metrics import bootstrap_intervals, classification_metrics


def test_confusion_matrix_orientation():
    metrics = classification_metrics([1, 1, 0, 0], [1, 0, 1, 0])
    assert (metrics["tp"], metrics["tn"], metrics["fp"], metrics["fn"]) == (1, 1, 1, 1)
    assert metrics["precision"] == pytest.approx(.5)
    assert metrics["recall"] == pytest.approx(.5)
    assert metrics["f1"] == pytest.approx(.5)


def test_precision_and_recall_are_not_swapped():
    metrics = classification_metrics([1, 1, 1, 0], [1, 0, 0, 0])
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == pytest.approx(1 / 3)


def test_bootstrap_is_reproducible():
    one = bootstrap_intervals([0, 0, 1, 1], [0, 1, 1, 1], iterations=20, seed=7)
    two = bootstrap_intervals([0, 0, 1, 1], [0, 1, 1, 1], iterations=20, seed=7)
    assert one == two

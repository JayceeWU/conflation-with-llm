import pandas as pd

from conflation_benchmark.data import create_manifest, validate_manifest


def test_component_split_has_no_entity_leakage_and_is_reproducible():
    frame = pd.DataFrame({
        "example_id": [f"e{i}" for i in range(8)], "source_index": range(8),
        "id": ["a", "b", "d", "f", "h", "j", "l", "n"],
        "base_id": ["b", "c", "e", "g", "i", "k", "m", "o"],
        "label": [1, 1, 0, 0, 1, 0, 1, 0],
    })
    ratios = {"train": .7, "validation": .15, "test": .15}
    first = create_manifest(frame, ratios, 42)
    second = create_manifest(frame, ratios, 42)
    pd.testing.assert_frame_equal(first, second)
    validate_manifest(first)
    assert first[first.id.isin(["a", "b"])].split.nunique() == 1

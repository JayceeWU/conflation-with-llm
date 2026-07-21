from conflation_benchmark.prompting import parse_label


def test_strict_prompt_parser():
    assert parse_label("MATCH") == (1, True)
    assert parse_label(" NO_MATCH\n") == (0, True)
    for invalid in ("", "match", "MATCH.", '{"label":"MATCH"}', "MATCH because names agree"):
        assert parse_label(invalid)[1] is False

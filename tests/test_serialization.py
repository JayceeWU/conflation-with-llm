from conflation_benchmark.serialization import serialize_pair


def sample():
    return {
        "names": '{"primary":"Cafe A"}', "base_names": '{"primary":"Cafe A"}',
        "categories": '{"primary":"cafe"}', "base_categories": '{"primary":"cafe"}',
        "websites": '["example.com"]', "base_websites": '["example.com"]',
        "socials": None, "base_socials": None, "emails": None, "base_emails": '["a@example.com"]',
        "phones": '["123"]', "base_phones": '["123"]', "brand": "{}", "base_brand": "{}",
        "addresses": '[{"country":"US"}]', "base_addresses": '[{"country":"US"}]',
        "id": "secret-a", "base_id": "secret-b", "sources": "secret-source", "confidence": .9,
    }


def test_stable_serialization_and_no_leaky_columns():
    first = serialize_pair(sample())
    assert first == serialize_pair(sample())
    assert "secret-a" not in first and "secret-source" not in first
    assert "[PAIR_SEPARATOR]" in first


def test_ablation_removes_both_sides_only():
    text = serialize_pair(sample(), "no_email")
    assert "EMAIL:" not in text
    assert "NAME:" in text and "WEBSITE:" in text

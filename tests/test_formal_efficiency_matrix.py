from scripts.run_formal_efficiency_matrix import (
    FORMAL_CONFIG_IDS,
    build_schedule,
    caption_hash,
    metric_summary,
)


def test_formal_schedule_is_deterministic_and_complete():
    schedule = build_schedule()
    assert schedule == build_schedule()
    assert len(schedule["blocks"]) == 3
    assert schedule["batch_order"] == [1, 2, 4, 8, 16]
    for block in schedule["blocks"]:
        assert sorted(block["configuration_order"]) == FORMAL_CONFIG_IDS


def test_metric_summary_reports_median_and_iqr():
    result = metric_summary([1.0, 2.0, 3.0, 4.0])
    assert result["count"] == 4
    assert result["median"] == 2.5
    assert result["q25"] == 1.75
    assert result["q75"] == 3.25
    assert result["iqr"] == 1.5


def test_caption_hash_is_order_sensitive_and_stable():
    assert caption_hash(["a", "b"]) == caption_hash(["a", "b"])
    assert caption_hash(["a", "b"]) != caption_hash(["b", "a"])

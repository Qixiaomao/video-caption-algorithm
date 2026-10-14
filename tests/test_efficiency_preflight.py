from scripts.freeze_efficiency_inputs import select_records, selection_digest
from scripts.run_efficiency_preflight import configurations


def test_input_selection_is_hash_ranked_and_stable():
    records = [{"video_id": value} for value in ("v3", "v1", "v4", "v2")]
    selected = select_records(records, size=3)
    expected = sorted(records, key=lambda row: (selection_digest(row["video_id"]), row["video_id"]))[:3]
    assert selected == expected


def test_efficiency_matrix_is_frozen_and_incremental():
    configs = configurations()
    assert [row["id"] for row in configs] == ["E0", "E1", "E2", "E3", "E4", "E5"]
    assert [row["required"] for row in configs] == [True, True, True, True, True, False]
    assert not any(config.get("residual_layernorm", False) for config in configs)
    assert configs[0]["fp16"] is False
    assert configs[-1]["compile"] is True

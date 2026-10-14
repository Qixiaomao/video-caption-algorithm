import numpy as np

from scripts.summarize_efficiency_uncertainty import (
    contrast_value,
    derived_seed,
    hierarchical_contrast_interval,
    hierarchical_median_interval,
)


def test_derived_seed_is_stable_and_label_specific():
    assert derived_seed("a") == derived_seed("a")
    assert derived_seed("a") != derived_seed("b")


def test_hierarchical_absolute_interval_is_reproducible():
    blocks = {
        1: np.asarray([1.0, 1.0, 1.0]),
        2: np.asarray([2.0, 2.0, 2.0]),
        3: np.asarray([3.0, 3.0, 3.0]),
    }
    first = hierarchical_median_interval(blocks, label="test", resamples=200)
    second = hierarchical_median_interval(blocks, label="test", resamples=200)
    assert first == second
    assert first["point"] == 2.0
    assert first["ci95_low"] <= 2.0 <= first["ci95_high"]


def test_paired_hierarchical_contrast_detects_uniform_improvement():
    reference = {block: np.asarray([10.0, 10.0, 10.0]) for block in (1, 2, 3)}
    candidate = {block: np.asarray([8.0, 8.0, 8.0]) for block in (1, 2, 3)}
    result = hierarchical_contrast_interval(
        reference,
        candidate,
        effect="latency_reduction_percent",
        label="uniform",
        resamples=200,
    )
    assert result["point"] == 20.0
    assert result["ci95_low"] == 20.0
    assert result["ci95_high"] == 20.0
    assert result["directionally_supported"] is True


def test_contrast_directions_are_positive_when_candidate_is_better():
    assert contrast_value(10.0, 8.0, "latency_reduction_percent") == 20.0
    assert contrast_value(2.0, 3.0, "throughput_improvement_percent") == 50.0
    assert contrast_value(100.0, 80.0, "memory_reduction_mib") == 20.0

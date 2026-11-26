"""
Unit tests for causal post-processing functions.

Tests:
- Causality: changing future scores (t > t0) does not alter smoothed scores or flags for t <= t0.
- smooth_scores: causal trailing window mean and median correctness, edge cases.
- hysteresis_flags: stateful switching on > high and off < low, holding in between.
- min_duration_filter: filtering transient spikes, flagging only after m consecutive windows.
- apply_alerting: end-to-end alerting pipeline and causality.
- rederive_threshold: re-deriving threshold on smoothed validation distribution.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.postprocess import (
    apply_alerting,
    hysteresis_flags,
    min_duration_filter,
    rederive_threshold,
    smooth_scores,
)


def test_causality_smooth_scores():
    """Modifying future scores must NEVER alter past smoothed scores."""
    np.random.seed(42)
    scores = np.random.uniform(0.05, 0.5, size=50)
    t0 = 25

    smoothed_orig = smooth_scores(scores, k=5, method="mean")

    # Change future scores after t0
    scores_perturbed = scores.copy()
    scores_perturbed[t0 + 1 :] = np.random.uniform(1.0, 5.0, size=len(scores) - t0 - 1)

    smoothed_perturbed = smooth_scores(scores_perturbed, k=5, method="mean")

    # Assert outputs up to t0 are byte-for-byte identical
    np.testing.assert_array_equal(smoothed_orig[: t0 + 1], smoothed_perturbed[: t0 + 1])


def test_causality_apply_alerting():
    """Modifying future scores must NEVER alter past alert flags."""
    np.random.seed(42)
    scores = np.random.uniform(0.05, 0.5, size=60)
    t0 = 30
    threshold = 0.3

    flags_orig = apply_alerting(scores, threshold=threshold, k=5, m=3, low_ratio=0.8)

    # Change future scores completely
    scores_perturbed = scores.copy()
    scores_perturbed[t0 + 1 :] = 10.0

    flags_perturbed = apply_alerting(scores_perturbed, threshold=threshold, k=5, m=3, low_ratio=0.8)

    # Assert past flags up to t0 are completely unaffected
    np.testing.assert_array_equal(flags_orig[: t0 + 1], flags_perturbed[: t0 + 1])


def test_smooth_scores_trailing_mean():
    """Test exact numerical trailing mean."""
    scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # k=3:
    # t=0: [1.0] -> 1.0
    # t=1: [1.0, 2.0] -> 1.5
    # t=2: [1.0, 2.0, 3.0] -> 2.0
    # t=3: [2.0, 3.0, 4.0] -> 3.0
    # t=4: [3.0, 4.0, 5.0] -> 4.0
    expected = np.array([1.0, 1.5, 2.0, 3.0, 4.0])
    res = smooth_scores(scores, k=3, method="mean")
    np.testing.assert_allclose(res, expected)


def test_smooth_scores_trailing_median():
    """Test exact numerical trailing median."""
    scores = np.array([1.0, 10.0, 2.0, 3.0, 100.0])
    # k=3:
    # t=0: [1.0] -> 1.0
    # t=1: [1.0, 10.0] -> 5.5
    # t=2: [1.0, 10.0, 2.0] -> 2.0
    # t=3: [10.0, 2.0, 3.0] -> 3.0
    # t=4: [2.0, 3.0, 100.0] -> 3.0
    expected = np.array([1.0, 5.5, 2.0, 3.0, 3.0])
    res = smooth_scores(scores, k=3, method="median")
    np.testing.assert_allclose(res, expected)


def test_smooth_scores_k1():
    """k=1 returns identical copy."""
    scores = np.array([0.1, 0.4, 0.2])
    res = smooth_scores(scores, k=1)
    np.testing.assert_array_equal(res, scores)


def test_hysteresis_flags():
    """Test hysteresis behavior: turns on > high, turns off < low, holds in between."""
    high = 1.0
    low = 0.5
    scores = np.array([
        0.2,  # < low, state 0
        0.8,  # between, state remains 0
        1.1,  # > high, state turns 1
        0.9,  # between, state remains 1
        0.6,  # between, state remains 1
        0.4,  # < low, state turns 0
        0.7,  # between, state remains 0
        1.2,  # > high, state turns 1
        0.3,  # < low, state turns 0
    ])
    expected = np.array([0, 0, 1, 1, 1, 0, 0, 1, 0])
    res = hysteresis_flags(scores, high=high, low=low)
    np.testing.assert_array_equal(res, expected)


def test_min_duration_filter():
    """Test minimum duration filtering: flags only after m consecutive 1s."""
    m = 3
    # Spikes of length 1, 2, and 4
    flags = np.array([0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1])
    # Expected:
    # idx 0: 0
    # idx 1: 1 (run 1 < 3) -> 0
    # idx 2: 0 -> 0
    # idx 3: 1 (run 1 < 3) -> 0
    # idx 4: 1 (run 2 < 3) -> 0
    # idx 5: 0 -> 0
    # idx 6: 1 (run 1 < 3) -> 0
    # idx 7: 1 (run 2 < 3) -> 0
    # idx 8: 1 (run 3 >= 3) -> 1
    # idx 9: 1 (run 4 >= 3) -> 1
    # idx 10: 0 -> 0
    # idx 11: 1 (run 1 < 3) -> 0
    # idx 12: 1 (run 2 < 3) -> 0
    # idx 13: 1 (run 3 >= 3) -> 1
    expected = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1])
    res = min_duration_filter(flags, m=m)
    np.testing.assert_array_equal(res, expected)


def test_min_duration_filter_m1():
    """m=1 returns identical flags."""
    flags = np.array([0, 1, 0, 1, 1, 0])
    res = min_duration_filter(flags, m=1)
    np.testing.assert_array_equal(res, flags)


def test_rederive_threshold():
    """Smoothing reduces variance, so p99 threshold on smoothed val scores is lower/calibrated."""
    np.random.seed(42)
    val_scores = np.random.exponential(scale=0.1, size=5000)
    raw_p99 = np.percentile(val_scores, 99.0)
    smoothed_p99 = rederive_threshold(val_scores, k=5, percentile=99.0, method="mean")
    # Smoothed distribution should have lower tail percentile than noisy raw
    assert smoothed_p99 < raw_p99

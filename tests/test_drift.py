"""Tests for the Population Stability Index (PSI) and the drift report. Synthetic data only."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stroke_model.reference import (
    bin_shares,
    categorical_psi,
    make_inner_edges,
    psi_from_shares,
    psi_status,
)

REFERENCE_PATH = Path(__file__).resolve().parent.parent / "monitoring" / "reference_stats.json"


def test_psi_is_zero_for_identical_distributions():
    shares = [0.1] * 10
    assert psi_from_shares(shares, shares) == pytest.approx(0.0, abs=1e-9)


def test_psi_matches_hand_calculation():
    """Two bins: reference 50/50, new 75/25. PSI = 0.25*ln(1.5) + (-0.25)*ln(0.5)."""
    expected = 0.25 * np.log(1.5) + (-0.25) * np.log(0.5)
    assert psi_from_shares([0.5, 0.5], [0.75, 0.25]) == pytest.approx(expected, abs=1e-3)


def test_bin_shares_sum_to_one_and_count_out_of_range_values():
    rng = np.random.default_rng(0)
    reference = rng.normal(70, 10, 2000)
    edges = make_inner_edges(reference)

    # Values far outside the training range land in the first and last bins
    shares = bin_shares([-1000.0, 1000.0], edges)
    assert sum(shares) == pytest.approx(1.0)
    assert shares[0] == pytest.approx(0.5)
    assert shares[-1] == pytest.approx(0.5)


def test_psi_status_thresholds():
    assert psi_status(0.05) == "STABLE"
    assert psi_status(0.15) == "MONITOR"
    assert psi_status(0.25) == "DRIFT_ALERT"


def test_psi_detects_shift_in_numeric_distribution():
    rng = np.random.default_rng(1)
    reference = rng.normal(70, 10, 5000)
    edges = make_inner_edges(reference)
    ref_shares = bin_shares(reference, edges)

    same = psi_from_shares(ref_shares, bin_shares(rng.normal(70, 10, 5000), edges))
    shifted = psi_from_shares(ref_shares, bin_shares(rng.normal(80, 10, 5000), edges))

    assert same < 0.1
    assert shifted >= 0.2


def test_categorical_psi_handles_unseen_category():
    reference = {"M": 0.5, "F": 0.5}
    stable = categorical_psi(reference, pd.Series(["M", "F"] * 100))
    new_category = categorical_psi(reference, pd.Series(["M", "F", "X"] * 100))
    assert stable < 0.1
    assert new_category > stable


def test_reference_stats_contain_psi_bins():
    with open(REFERENCE_PATH) as f:
        ref = json.load(f)
    for col in ["age", "sbp", "delay"]:
        shares = ref["numeric"][col]["psi_bin_shares"]
        edges = ref["numeric"][col]["psi_inner_edges"]
        assert len(shares) == len(edges) + 1
        assert sum(shares) == pytest.approx(1.0)


def test_drift_report_flags_spread_change_that_mean_shift_misses():
    """Every patient has exactly the training mean age. The mean has not moved, but the
    distribution has collapsed to a point, which PSI should flag."""
    import sys
    sys.path.insert(0, str(REFERENCE_PATH.parent))
    from drift_check import run_drift_analysis

    with open(REFERENCE_PATH) as f:
        ref_mean_age = json.load(f)["numeric"]["age"]["mean"]

    production = pd.DataFrame({"age": [ref_mean_age] * 200})
    report = run_drift_analysis(production, REFERENCE_PATH)["numeric_shifts"]["age"]

    assert report["mean_shift_status"] == "STABLE"
    assert report["psi_status"] == "DRIFT_ALERT"

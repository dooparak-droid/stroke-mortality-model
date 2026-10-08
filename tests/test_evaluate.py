"""Tests for the bootstrap confidence intervals on the holdout metrics. Synthetic data only."""

import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import brier_score_loss, roc_auc_score

from stroke_model.evaluate import bootstrap_metric_intervals

METADATA_PATH = Path(__file__).resolve().parent.parent / "models" / "model_metadata.json"


def _synthetic_holdout(seed: int = 0, n: int = 1500, event_rate: float = 0.05):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < event_rate).astype(int)
    prob = np.clip(0.05 + 0.15 * y + rng.normal(0, 0.05, n), 0.001, 0.999)
    return y, prob


def test_interval_contains_point_estimate_and_is_ordered():
    y, prob = _synthetic_holdout()
    iv = bootstrap_metric_intervals(y, prob, n_resamples=300, seed=1)

    auc = roc_auc_score(y, prob)
    brier = brier_score_loss(y, prob)
    assert iv["auc"]["lower"] < iv["auc"]["upper"]
    assert iv["auc"]["lower"] <= auc <= iv["auc"]["upper"]
    assert iv["brier_score"]["lower"] < iv["brier_score"]["upper"]
    assert iv["brier_score"]["lower"] <= brier <= iv["brier_score"]["upper"]


def test_interval_is_reproducible_with_fixed_seed():
    y, prob = _synthetic_holdout()
    first = bootstrap_metric_intervals(y, prob, n_resamples=200, seed=7)
    second = bootstrap_metric_intervals(y, prob, n_resamples=200, seed=7)
    assert first == second


def test_interval_works_with_very_few_deaths():
    """Stratified resampling must keep deaths in every resample, so AUC is always defined."""
    y = np.zeros(200, dtype=int)
    y[:3] = 1
    prob = np.linspace(0.9, 0.01, 200)
    iv = bootstrap_metric_intervals(y, prob, n_resamples=200, seed=3)
    assert 0.0 <= iv["auc"]["lower"] <= iv["auc"]["upper"] <= 1.0


def test_saved_intervals_contain_saved_test_metrics():
    """Intervals stored in the model metadata must bracket the stored point estimates."""
    with open(METADATA_PATH) as f:
        meta = json.load(f)
    iv = meta["test_metric_intervals"]
    metrics = meta["test_metrics"]

    assert iv["auc"]["lower"] < iv["auc"]["upper"]
    assert iv["auc"]["lower"] <= metrics["auc"] <= iv["auc"]["upper"]
    assert iv["brier_score"]["lower"] < iv["brier_score"]["upper"]
    assert iv["brier_score"]["lower"] <= metrics["brier_score"] <= iv["brier_score"]["upper"]


# --- Calibration by subgroup -------------------------------------------------

import pandas as pd

from stroke_model.evaluate import (
    save_subgroup_calibration_report,
    subgroup_calibration,
    wilson_interval,
)


def _synthetic_features(n: int = 600, seed: int = 5):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "age": rng.integers(30, 95, n),
        "gender": rng.choice(["F", "M"], n),
        "subtype": rng.choice(["LACS", "PACS", "POCS", "TACS", "OTH"], n),
    })


def test_wilson_interval_known_values():
    lower, upper = wilson_interval(5, 10)
    assert lower + upper == pytest.approx(1.0, abs=1e-9)   # symmetric around 0.5
    assert lower < 0.5 < upper

    lower, upper = wilson_interval(0, 12)                   # no deaths in a small group
    assert lower == 0.0
    assert upper == pytest.approx(0.2425, abs=1e-3)


def test_subgroup_counts_cover_every_patient_once():
    features = _synthetic_features()
    rng = np.random.default_rng(6)
    y = (rng.random(len(features)) < 0.05).astype(int)
    prob = np.clip(rng.normal(0.05, 0.03, len(features)), 0.001, 0.99)

    table = subgroup_calibration(features, y, prob)
    assert set(table) == {"age_band", "gender", "subtype"}
    for rows in table.values():
        assert sum(r["patients"] for r in rows) == len(features)
        assert sum(r["deaths"] for r in rows) == int(y.sum())


def test_subgroup_observed_rate_lies_inside_its_interval():
    features = _synthetic_features()
    rng = np.random.default_rng(7)
    y = (rng.random(len(features)) < 0.08).astype(int)
    prob = np.full(len(features), 0.08)

    for rows in subgroup_calibration(features, y, prob).values():
        for r in rows:
            if r["patients"] > 0:
                assert r["observed_ci_lower"] <= r["observed"] <= r["observed_ci_upper"]


def test_subgroup_report_files_are_written(tmp_path):
    features = _synthetic_features()
    rng = np.random.default_rng(8)
    y = (rng.random(len(features)) < 0.05).astype(int)
    prob = np.clip(rng.normal(0.05, 0.03, len(features)), 0.001, 0.99)

    save_subgroup_calibration_report(features, y, prob, tmp_path)
    assert (tmp_path / "subgroup_calibration.png").stat().st_size > 0
    saved = json.loads((tmp_path / "subgroup_calibration.json").read_text())
    assert saved["gender"][0]["group"] == "F"

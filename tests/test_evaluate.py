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

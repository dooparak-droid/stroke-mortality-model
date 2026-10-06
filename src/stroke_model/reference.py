"""Reference distributions and Population Stability Index (PSI) for drift monitoring.

PSI compares how a variable is distributed in two groups of patients, here the training
patients (the reference) and a batch of new patients. The reference is stored as bin edges
and the share of training patients in each bin, so the raw training data is never needed
at monitoring time.
"""

from typing import Dict, List, Sequence
import numpy as np
import pandas as pd

NUM_BINS = 10
SMOOTHING = 1e-4

# PSI interpretation thresholds (conventional rule of thumb)
PSI_MONITOR = 0.1
PSI_ALERT = 0.2

# PSI is noisy when it is calculated from only a few records
MIN_RELIABLE_SAMPLE = 100


def make_inner_edges(values: Sequence[float], num_bins: int = NUM_BINS) -> List[float]:
    """Return the interior bin edges (quantiles) of the reference values.

    Values below the first edge fall in the first bin and values above the last edge fall
    in the last bin, so new data outside the training range is still counted. Repeated
    quantiles (common for integer-valued variables) are merged, so there can be fewer bins
    than requested.
    """
    percentiles = np.linspace(0, 100, num_bins + 1)[1:-1]
    edges = np.unique(np.percentile(np.asarray(values, dtype=float), percentiles))
    return [float(e) for e in edges]


def bin_shares(values: Sequence[float], inner_edges: Sequence[float]) -> List[float]:
    """Share of values falling in each bin defined by the interior edges."""
    index = np.searchsorted(np.asarray(inner_edges), np.asarray(values, dtype=float), side="right")
    counts = np.bincount(index, minlength=len(inner_edges) + 1)
    return [float(c) for c in counts / counts.sum()]


def psi_from_shares(expected: Sequence[float], actual: Sequence[float]) -> float:
    """Population Stability Index from two lists of bin shares.

    PSI = sum over bins of (actual - expected) * ln(actual / expected). A small constant
    is added to every share so that an empty bin does not cause a division by zero.
    """
    exp = np.asarray(expected, dtype=float) + SMOOTHING
    act = np.asarray(actual, dtype=float) + SMOOTHING
    exp = exp / exp.sum()
    act = act / act.sum()
    return float(np.sum((act - exp) * np.log(act / exp)))


def psi_status(psi: float) -> str:
    """Label a PSI value using the conventional thresholds."""
    if psi >= PSI_ALERT:
        return "DRIFT_ALERT"
    if psi >= PSI_MONITOR:
        return "MONITOR"
    return "STABLE"


def numeric_reference(values: pd.Series) -> Dict[str, object]:
    """Summary statistics and PSI bins for one numeric predictor in the training data."""
    edges = make_inner_edges(values.values)
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
        "median": float(values.median()),
        "q25": float(values.quantile(0.25)),
        "q75": float(values.quantile(0.75)),
        "psi_inner_edges": edges,
        "psi_bin_shares": bin_shares(values.values, edges),
    }


def categorical_psi(reference: Dict[str, float], production: pd.Series) -> float:
    """PSI for a categorical predictor, using category shares as the bins."""
    categories = sorted(set(reference) | set(production.unique()), key=str)
    prod_shares = production.value_counts(normalize=True).to_dict()
    expected = [reference.get(c, 0.0) for c in categories]
    actual = [prod_shares.get(c, 0.0) for c in categories]
    return psi_from_shares(expected, actual)

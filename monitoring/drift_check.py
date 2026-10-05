"""Input drift detection comparing inference requests against training distributions."""

import json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd


def calculate_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    num_bins: int = 10
) -> float:
    """Calculate Population Stability Index (PSI) between reference and production data.

    PSI Interpretation:
        PSI < 0.1: No significant change / stable
        0.1 <= PSI < 0.2: Moderate shift / monitor closely
        PSI >= 0.2: Significant drift / investigate and consider retraining

    Args:
        expected: Reference values from training set.
        actual: Production/inference values.
        num_bins: Number of quantiles for binning.

    Returns:
        Calculated PSI value.
    """
    if len(actual) == 0 or len(expected) == 0:
        return 0.0

    # Determine quantile bins based on reference data
    percentiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(expected, percentiles)
    bin_edges = np.unique(bin_edges)  # Avoid duplicate edges

    if len(bin_edges) < 2:
        return 0.0

    # Count occurrences in bins
    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)

    # Convert to fractions with Laplace smoothing to avoid division by zero
    expected_pct = (expected_counts + 1e-4) / (len(expected) + 1e-4 * len(expected_counts))
    actual_pct = (actual_counts + 1e-4) / (len(actual) + 1e-4 * len(actual_counts))

    psi_value = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi_value)


def run_drift_analysis(
    production_df: pd.DataFrame,
    reference_path: str | Path = "monitoring/reference_stats.json"
) -> Dict[str, Any]:
    """Run drift analysis on numeric and categorical features against stored reference stats.

    Args:
        production_df: DataFrame of incoming inference requests.
        reference_path: Path to reference_stats.json.

    Returns:
        Dictionary summarizing drift scores per feature.
    """
    ref_file = Path(reference_path)
    if not ref_file.is_file():
        raise FileNotFoundError(f"Reference statistics not found at: {ref_file}")

    with open(ref_file, "r") as f:
        ref_stats = json.load(f)

    results: Dict[str, Any] = {
        "sample_size": len(production_df),
        "numeric_shifts": {},
        "categorical_shifts": {}
    }

    # Check numeric feature drift (mean shift in standard deviations)
    for col in ["age", "sbp", "delay"]:
        if col in production_df.columns:
            ref_mean = ref_stats["numeric"][col]["mean"]
            ref_std = ref_stats["numeric"][col]["std"]
            prod_mean = float(production_df[col].mean())
            z_shift = (prod_mean - ref_mean) / ref_std if ref_std > 0 else 0.0

            results["numeric_shifts"][col] = {
                "reference_mean": round(ref_mean, 2),
                "production_mean": round(prod_mean, 2),
                "z_score_shift": round(z_shift, 3),
                "status": "DRIFT_ALERT" if abs(z_shift) >= 0.5 else "STABLE"
            }

    # Check categorical feature distribution differences
    for col, ref_dist in ref_stats.get("categorical", {}).items():
        if col in production_df.columns:
            prod_dist = production_df[col].value_counts(normalize=True).to_dict()
            max_diff = max(
                abs(prod_dist.get(cat, 0.0) - ref_dist.get(cat, 0.0))
                for cat in set(ref_dist.keys()) | set(prod_dist.keys())
            )
            results["categorical_shifts"][col] = {
                "max_percentage_point_difference": round(max_diff * 100, 2),
                "status": "DRIFT_ALERT" if max_diff >= 0.15 else "STABLE"
            }

    return results


if __name__ == "__main__":
    import sys
    print("=== Stroke Model Drift Check ===")
    ref_json = Path(__file__).parent / "reference_stats.json"

    if len(sys.argv) > 1:
        log_csv = sys.argv[1]
        prod_data = pd.read_csv(log_csv)
    else:
        print("No log CSV provided. Generating synthetic production sample for demonstration...")
        # Create a small sample mimicking a slightly older cohort
        np.random.seed(42)
        n = 100
        prod_data = pd.DataFrame({
            "age": np.random.normal(73, 10, n),
            "sbp": np.random.normal(162, 25, n),
            "delay": np.random.uniform(5, 30, n),
            "gender": np.random.choice(["M", "F"], n),
            "consc": np.random.choice(["F", "D"], n, p=[0.85, 0.15])
        })

    report = run_drift_analysis(prod_data, ref_json)
    print(json.dumps(report, indent=2))

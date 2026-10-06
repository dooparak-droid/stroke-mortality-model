"""Input drift detection comparing incoming records against training distributions.

For each numeric predictor the report gives the Population Stability Index (PSI) and the
shift in the mean in training standard deviations. For each categorical predictor it gives
the PSI and the largest change in any category share. See stroke_model/reference.py.
"""

import json
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd

from stroke_model.reference import (
    MIN_RELIABLE_SAMPLE,
    bin_shares,
    categorical_psi,
    psi_from_shares,
    psi_status,
)

NUMERIC_PREDICTORS = ["age", "sbp", "delay"]
MEAN_SHIFT_ALERT = 0.5  # training standard deviations
CATEGORY_SHIFT_ALERT = 0.15  # absolute change in a category share


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
    if len(production_df) < MIN_RELIABLE_SAMPLE:
        results["note"] = (
            f"Only {len(production_df)} records. PSI is noisy for small samples, "
            f"so treat alerts as provisional."
        )

    # Numeric predictors: PSI over training-quantile bins, plus mean shift
    for col in NUMERIC_PREDICTORS:
        if col in production_df.columns:
            ref = ref_stats["numeric"][col]
            values = production_df[col].dropna().values
            psi = psi_from_shares(
                ref["psi_bin_shares"], bin_shares(values, ref["psi_inner_edges"])
            )
            ref_std = ref["std"]
            prod_mean = float(np.mean(values))
            z_shift = (prod_mean - ref["mean"]) / ref_std if ref_std > 0 else 0.0

            results["numeric_shifts"][col] = {
                "psi": round(psi, 4),
                "psi_status": psi_status(psi),
                "reference_mean": round(ref["mean"], 2),
                "production_mean": round(prod_mean, 2),
                "z_score_shift": round(z_shift, 3),
                "mean_shift_status": (
                    "DRIFT_ALERT" if abs(z_shift) >= MEAN_SHIFT_ALERT else "STABLE"
                ),
            }

    # Categorical predictors: PSI over category shares, plus largest share change
    for col, ref_dist in ref_stats.get("categorical", {}).items():
        if col in production_df.columns:
            values = production_df[col].dropna()
            psi = categorical_psi(ref_dist, values)
            prod_dist = values.value_counts(normalize=True).to_dict()
            max_diff = max(
                abs(prod_dist.get(cat, 0.0) - ref_dist.get(cat, 0.0))
                for cat in set(ref_dist.keys()) | set(prod_dist.keys())
            )
            results["categorical_shifts"][col] = {
                "psi": round(psi, 4),
                "psi_status": psi_status(psi),
                "max_percentage_point_difference": round(max_diff * 100, 2),
                "share_shift_status": (
                    "DRIFT_ALERT" if max_diff >= CATEGORY_SHIFT_ALERT else "STABLE"
                ),
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
        # Synthetic sample of a slightly different cohort (older, higher blood pressure)
        rng = np.random.default_rng(42)
        n = 500
        prod_data = pd.DataFrame({
            "age": rng.normal(76, 10, n).round(),
            "sbp": rng.normal(168, 25, n).round(),
            "delay": rng.uniform(5, 30, n).round(),
            "gender": rng.choice(["M", "F"], n),
            "consc": rng.choice(["F", "D"], n, p=[0.85, 0.15])
        })

    report = run_drift_analysis(prod_data, ref_json)
    print(json.dumps(report, indent=2))

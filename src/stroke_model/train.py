"""Training pipeline for Ridge logistic regression with cross-validation and calibration."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from stroke_model.data import load_raw_data, split_cohort_data, TARGET_COLUMN
from stroke_model.features import ALL_PREDICTORS, create_preprocessor
from stroke_model.reference import numeric_reference
from stroke_model.evaluate import (
    bootstrap_metric_intervals,
    compute_youden_threshold,
    evaluate_predictions,
    plot_calibration,
    save_subgroup_calibration_report
)


def train_ridge_model(
    data_path: str | Path,
    output_dir: str | Path = "models",
    reports_dir: str | Path = "reports"
) -> Dict[str, Any]:
    """Execute complete model training and validation pipeline.

    Args:
        data_path: Path to raw dataset CSV.
        output_dir: Directory to save serialised model and metadata.
        reports_dir: Directory to export calibration curves and metric tables.

    Returns:
        Dictionary of final test performance metrics and artifact metadata.
    """
    output_path = Path(output_dir)
    reports_path = Path(reports_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    reports_path.mkdir(parents=True, exist_ok=True)

    print("Step 1: Loading cohort dataset...")
    df = load_raw_data(data_path)

    print("Step 2: Stratified train/test splitting (75% train, 25% test)...")
    train_df, test_df = split_cohort_data(df, test_size=0.25, random_state=123)

    X_train = train_df[ALL_PREDICTORS]
    y_train = train_df[TARGET_COLUMN].values
    X_test = test_df[ALL_PREDICTORS]
    y_test = test_df[TARGET_COLUMN].values

    # No class weighting: each patient counts equally so that predicted
    # probabilities stay on the scale of the observed death rate. The low event
    # rate is handled by the Youden threshold chosen in Step 5.
    print(f"  Training deaths: {int(np.sum(y_train == 1))} of {len(y_train)} ({np.mean(y_train):.1%})")

    print("Step 3: Setting up preprocessing and Ridge logistic regression pipeline...")
    pipeline = Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=1000,
                    random_state=123
                )
            )
        ]
    )

    print("Step 4: Hyperparameter tuning via 10-fold Stratified CV (optimizing ROC AUC)...")
    param_grid = {
        "classifier__C": np.logspace(-4, 2, 20)
    }

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    best_c = float(grid_search.best_params_["classifier__C"])
    best_cv_auc = float(grid_search.best_score_)
    print(f"  Optimal C: {best_c:.5f} (CV AUC: {best_cv_auc:.4f})")

    best_pipeline = grid_search.best_estimator_

    print("Step 5: Optimising Youden threshold using out-of-fold CV predictions...")
    oof_probs = cross_val_predict(
        best_pipeline,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1
    )[:, 1]

    optimal_threshold, oof_sens, oof_spec = compute_youden_threshold(y_train, oof_probs)
    print(
        f"  Youden threshold: {optimal_threshold:.4f} "
        f"(OOF Sensitivity: {oof_sens:.4f}, Specificity: {oof_spec:.4f})"
    )

    print("Step 6: Fitting final pipeline on entire training partition...")
    best_pipeline.fit(X_train, y_train)

    print("Step 7: Evaluating on untouched holdout test partition...")
    test_probs = best_pipeline.predict_proba(X_test)[:, 1]
    test_metrics = evaluate_predictions(y_test, test_probs, threshold=optimal_threshold)

    print(f"  Test AUC: {test_metrics['auc']:.4f}")
    print(f"  Test Brier Score: {test_metrics['brier_score']:.4f}")
    print(f"  Test Sensitivity: {test_metrics['sensitivity']:.4f}")
    print(f"  Test Specificity: {test_metrics['specificity']:.4f}")

    print("  Bootstrapping 95% intervals for test AUC and Brier score (2,000 resamples)...")
    test_intervals = bootstrap_metric_intervals(y_test, test_probs)
    print(f"  Test AUC 95% CI: {test_intervals['auc']['lower']:.4f} to {test_intervals['auc']['upper']:.4f}")
    print(f"  Test Brier 95% CI: {test_intervals['brier_score']['lower']:.4f} to {test_intervals['brier_score']['upper']:.4f}")

    # Generate calibration plot
    cal_plot_path = reports_path / "calibration_plot.png"
    plot_calibration(y_test, test_probs, cal_plot_path)
    print(f"  Calibration plot saved to: {cal_plot_path}")

    save_subgroup_calibration_report(X_test, y_test, test_probs, reports_path)
    print(f"  Subgroup calibration report saved to: {reports_path}")

    # Export reference statistics for drift monitoring
    print("Step 8: Exporting training reference distributions for monitoring...")
    numeric_stats = {col: numeric_reference(train_df[col]) for col in ["age", "sbp", "delay"]}
    categorical_stats = {
        col: train_df[col].value_counts(normalize=True).to_dict()
        for col in ALL_PREDICTORS if col not in ["age", "sbp", "delay"]
    }
    ref_stats = {
        "numeric": numeric_stats,
        "categorical": categorical_stats,
        "sample_size": len(train_df)
    }

    monitoring_dir = Path("monitoring")
    monitoring_dir.mkdir(parents=True, exist_ok=True)
    with open(monitoring_dir / "reference_stats.json", "w") as f:
        json.dump(ref_stats, f, indent=2)

    # Save model artifact and metadata
    print("Step 9: Packaging model artifact and metadata...")
    model_version = "v2.0.0"
    model_file = output_path / "stroke_ridge_pipeline.joblib"
    joblib.dump(best_pipeline, model_file)

    metadata = {
        "model_version": model_version,
        "model_type": "Ridge Logistic Regression",
        "training_date": datetime.now(timezone.utc).isoformat(),
        "optimal_threshold": round(optimal_threshold, 4),
        "best_hyperparameter_C": best_c,
        "cv_auc": round(best_cv_auc, 4),
        "test_metrics": test_metrics,
        "predictors": ALL_PREDICTORS,
        "class_weighting": "none",
        "outcome_definition": "Death within 14 days of randomisation (International Stroke Trial indicator ID14), including deaths not recorded on the discharge form",
        "n_training_patients": len(train_df),
        "n_test_patients": len(test_df),
        "n_test_deaths": int(np.sum(y_test == 1)),
        "test_metric_intervals": test_intervals
    }

    metadata_file = output_path / "model_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Artifacts saved successfully:\n  - {model_file}\n  - {metadata_file}")
    return metadata


if __name__ == "__main__":
    import sys
    # Default to data in parent directory if not specified
    raw_path = sys.argv[1] if len(sys.argv) > 1 else "../ist_stroke_14day.csv"
    train_ridge_model(raw_path)

"""Model evaluation metrics: discrimination, calibration, and classification tables."""

from pathlib import Path
from typing import Dict, Any, Tuple
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)


def compute_youden_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray
) -> Tuple[float, float, float]:
    """Compute optimal classification threshold using Youden's J statistic.

    Youden's J = Sensitivity + Specificity - 1.

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_prob: Predicted probabilities of the positive class.

    Returns:
        Tuple of (optimal_threshold, sensitivity_at_threshold, specificity_at_threshold).
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    # Specificity = 1 - FPR
    j_scores = tpr + (1 - fpr) - 1
    best_idx = int(np.argmax(j_scores))

    optimal_threshold = float(thresholds[best_idx])
    sensitivity = float(tpr[best_idx])
    specificity = float(1.0 - fpr[best_idx])

    return optimal_threshold, sensitivity, specificity


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float
) -> Dict[str, Any]:
    """Calculate comprehensive evaluation metrics.

    Includes discrimination (AUC), calibration (Brier score), and diagnostic
    measures (sensitivity, specificity, precision, F1) at the chosen threshold.

    Args:
        y_true: Binary ground truth outcomes.
        y_prob: Predicted probabilities.
        threshold: Classification decision boundary.

    Returns:
        Dictionary of formatted evaluation metrics.
    """
    auc = float(roc_auc_score(y_true, y_prob))
    brier = float(brier_score_loss(y_true, y_prob))

    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    f1 = float(2 * precision * sensitivity / (precision + sensitivity)) if (precision + sensitivity) > 0 else 0.0

    return {
        "auc": round(auc, 4),
        "brier_score": round(brier, 4),
        "threshold": round(threshold, 4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "precision": round(precision, 4),
        "f1_score": round(f1, 4),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn)
    }


def plot_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: str | Path,
    n_bins: int = 10
) -> None:
    """Generate and save calibration plot comparing predicted vs observed mortality.

    Args:
        y_true: Ground truth binary labels.
        y_prob: Predicted mortality probabilities.
        output_path: File path to save the generated plot.
        n_bins: Number of probability bins.
    """
    prob_true, prob_pred = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="quantile"
    )

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    plt.plot(prob_pred, prob_true, marker="o", color="#2b5c8f", label="Model calibration")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Observed Proportion")
    plt.title("Calibration Curve (Quantile Bins)")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300)
    plt.close()

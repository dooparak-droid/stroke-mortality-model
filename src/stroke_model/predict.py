"""Inference module for loading model artifacts and generating mortality predictions."""

import json
from pathlib import Path
from typing import Dict, Any, Union
import joblib
import pandas as pd

from stroke_model.features import ALL_PREDICTORS

DEFAULT_MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"


class StrokePredictor:
    """Predictor wrapping the fitted Ridge pipeline and decision threshold."""

    def __init__(self, models_dir: Union[str, Path] = DEFAULT_MODELS_DIR):
        self.models_dir = Path(models_dir)
        self.model_path = self.models_dir / "stroke_ridge_pipeline.joblib"
        self.metadata_path = self.models_dir / "model_metadata.json"

        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Model artifact not found at {self.model_path}. "
                "Ensure training pipeline has executed."
            )

        self.pipeline = joblib.load(self.model_path)

        if self.metadata_path.is_file():
            with open(self.metadata_path, "r") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}

        self.threshold: float = float(self.metadata.get("optimal_threshold", 0.5))
        self.model_version: str = str(self.metadata.get("model_version", "unknown"))

    def predict_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Generate prediction for a single patient record.

        Args:
            record: Dictionary containing required clinical predictors.

        Returns:
            Dictionary containing predicted probability, binary decision,
            threshold used, model version, and clinical disclaimer.
        """
        df = pd.DataFrame([record])
        return self.predict_dataframe(df)[0]

    def predict_dataframe(self, df: pd.DataFrame) -> list[Dict[str, Any]]:
        """Generate predictions for a DataFrame of patient records.

        Args:
            df: DataFrame containing required clinical predictors.

        Returns:
            List of prediction dictionaries.
        """
        # Ensure all expected columns are present
        missing = set(ALL_PREDICTORS) - set(df.columns)
        if missing:
            raise ValueError(f"Input record missing required fields: {sorted(missing)}")

        probs = self.pipeline.predict_proba(df[ALL_PREDICTORS])[:, 1]

        results = []
        for prob in probs:
            p = float(prob)
            results.append({
                "mortality_probability": round(p, 4),
                "high_risk_flag": bool(p >= self.threshold),
                "threshold_applied": self.threshold,
                "model_version": self.model_version,
                "disclaimer": "Research demonstration only. Not for clinical use or medical decision-making."
            })
        return results

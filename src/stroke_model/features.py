"""Feature definitions and preprocessing pipeline specification."""

from typing import List
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES: List[str] = [
    "age",
    "sbp",
    "delay"
]

CATEGORICAL_FEATURES: List[str] = [
    "gender",
    "consc",
    "subtype",
    "treat1",
    "treat2",
    "wakesym",
    "atrial",
    "CT",
    "Infarc",
    "hep24",
    "asp3",
    "symptom1",
    "symptom2",
    "symptom3",
    "symptom4",
    "symptom5",
    "symptom6",
    "symptom7",
    "symptom8"
]

ALL_PREDICTORS: List[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def create_preprocessor() -> ColumnTransformer:
    """Construct column transformer for preprocessing raw tabular inputs.

    Numeric features are scaled using StandardScaler.
    Categorical features are dummy-encoded using OneHotEncoder with the first
    level dropped to avoid collinearity in regularised linear models.

    Returns:
        Configured ColumnTransformer instance.
    """
    numeric_transformer = StandardScaler()

    categorical_transformer = OneHotEncoder(
        drop="first",
        sparse_output=False,
        handle_unknown="ignore"
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES)
        ],
        remainder="drop"
    )

    return preprocessor

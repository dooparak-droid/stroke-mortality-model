"""Data loading and dataset splitting functions."""

from pathlib import Path
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

TARGET_COLUMN = "death"

EXPECTED_COLUMNS = [
    "delay", "consc", "gender", "age", "wakesym", "atrial", "CT",
    "Infarc", "hep24", "asp3", "sbp", "symptom1", "symptom2",
    "symptom3", "symptom4", "symptom5", "symptom6", "symptom7",
    "symptom8", "subtype", "treat1", "treat2", TARGET_COLUMN
]


def load_raw_data(filepath: str | Path) -> pd.DataFrame:
    """Load raw dataset from CSV and check column integrity.

    Args:
        filepath: Path to the raw CSV file.

    Returns:
        pd.DataFrame containing the cohort data.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If expected columns are missing.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Data file not found at: {path.resolve()}")

    df = pd.read_csv(path)

    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Dataset missing required columns: {sorted(missing_cols)}")

    return df


def split_cohort_data(
    df: pd.DataFrame,
    test_size: float = 0.25,
    random_state: int = 123
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Perform stratified split into training and holdout test sets.

    Args:
        df: Complete dataset.
        test_size: Fraction of cohort reserved for test evaluation (default 0.25).
        random_state: Fixed random seed for exact reproducibility (default 123).

    Returns:
        Tuple of (train_df, test_df).
    """
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[TARGET_COLUMN],
        random_state=random_state
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

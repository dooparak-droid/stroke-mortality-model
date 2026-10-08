"""Tests for the rules that build the training file from the IST database. Synthetic rows only."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "build_dataset.py"
spec = importlib.util.spec_from_file_location("build_dataset", SCRIPT)
build_dataset = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_dataset)


def _row(**overrides):
    """One synthetic IST-style row that passes every selection rule and survives."""
    row = {
        "RDELAY": 10, "RCONSC": "F", "SEX": "F", "AGE": 70, "RSLEEP": "N", "RATRIAL": "N",
        "RCT": "Y", "RVISINF": "N", "RHEP24": "N", "RASP3": "N", "RSBP": 150,
        "STYPE": "PACS", "RXHEP": "N", "RXASP": "N",
        "DDEAD": "N", "ID14": 0, "SET14D": 1,
    }
    row.update({f"RDEF{i}": "N" for i in range(1, 9)})
    row.update(overrides)
    return row


def test_outcome_comes_from_the_14_day_indicator():
    """A death by day 14 that the discharge form does not record still counts as a death."""
    ist = pd.DataFrame([
        _row(DDEAD="Y", ID14=1),   # died by day 14, recorded at discharge
        _row(DDEAD="N", ID14=1),   # died by day 14, not recorded at discharge
        _row(DDEAD="Y", ID14=0),   # recorded at discharge but died after day 14
        _row(DDEAD="N", ID14=0),   # survivor
    ])
    result = build_dataset.build_dataset(ist)
    assert result["death"].tolist() == [1, 1, 0, 0]


def test_selection_rules_exclude_the_expected_patients():
    ist = pd.DataFrame([
        _row(),                          # kept
        _row(RCONSC="U"),                # unconscious: excluded
        _row(RDEF3="C"),                 # deficit cannot be assessed: excluded
        _row(RATRIAL=None),              # missing predictor: excluded
        _row(DDEAD=None),                # unknown discharge-form status: excluded
        _row(DDEAD="U"),                 # unknown discharge-form status: excluded
    ])
    result = build_dataset.build_dataset(ist)
    assert len(result) == 1


def test_output_columns_use_model_names_and_end_with_outcome():
    result = build_dataset.build_dataset(pd.DataFrame([_row()]))
    assert list(result.columns) == build_dataset.OUTPUT_COLUMNS
    assert result.columns[-1] == "death"
    assert {"age", "sbp", "delay", "treat1", "treat2", "symptom8"} <= set(result.columns)


def test_unknown_14_day_status_stops_the_build():
    ist = pd.DataFrame([_row(SET14D=0)])
    with pytest.raises(ValueError):
        build_dataset.build_dataset(ist)

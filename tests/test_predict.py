"""Unit, regression, and clinical monotonicity tests for model prediction interface."""

import pytest
from stroke_model.predict import StrokePredictor

SYNTHETIC_RECORDS = [
    {
        "age": 55.0,
        "sbp": 130.0,
        "delay": 4.0,
        "gender": "M",
        "consc": "F",
        "subtype": "LACS",
        "treat1": "N",
        "treat2": "N",
        "wakesym": "N",
        "atrial": "N",
        "CT": "Y",
        "Infarc": "N",
        "hep24": "N",
        "asp3": "Y",
        "symptom1": "N",
        "symptom2": "N",
        "symptom3": "N",
        "symptom4": "N",
        "symptom5": "N",
        "symptom6": "N",
        "symptom7": "N",
        "symptom8": "N"
    },
    {
        "age": 88.0,
        "sbp": 185.0,
        "delay": 36.0,
        "gender": "F",
        "consc": "D",
        "subtype": "TACS",
        "treat1": "N",
        "treat2": "N",
        "wakesym": "Y",
        "atrial": "Y",
        "CT": "Y",
        "Infarc": "Y",
        "hep24": "N",
        "asp3": "N",
        "symptom1": "Y",
        "symptom2": "Y",
        "symptom3": "Y",
        "symptom4": "Y",
        "symptom5": "Y",
        "symptom6": "Y",
        "symptom7": "Y",
        "symptom8": "Y"
    }
]


def test_predictor_initialization():
    """Verify StrokePredictor initialises and loads metadata."""
    predictor = StrokePredictor()
    assert predictor.pipeline is not None
    assert 0.0 < predictor.threshold < 1.0
    assert predictor.model_version != "unknown"


def test_prediction_output_structure():
    """Verify single record prediction contains expected keys and probability bounds."""
    predictor = StrokePredictor()
    result = predictor.predict_record(SYNTHETIC_RECORDS[0])

    assert "mortality_probability" in result
    assert 0.0 <= result["mortality_probability"] <= 1.0
    assert "high_risk_flag" in result
    assert isinstance(result["high_risk_flag"], bool)
    assert result["threshold_applied"] == predictor.threshold


def test_risk_ordering_sanity():
    """Verify that a high-risk synthetic patient receives a higher probability than a low-risk patient."""
    predictor = StrokePredictor()
    low_risk = predictor.predict_record(SYNTHETIC_RECORDS[0])
    high_risk = predictor.predict_record(SYNTHETIC_RECORDS[1])

    assert high_risk["mortality_probability"] > low_risk["mortality_probability"]


def test_clinical_monotonicity_age():
    """Clinically: older age must yield higher predicted mortality, holding other factors constant."""
    predictor = StrokePredictor()
    base = dict(SYNTHETIC_RECORDS[0])

    younger = dict(base, age=50.0)
    older = dict(base, age=85.0)

    p_young = predictor.predict_record(younger)["mortality_probability"]
    p_old = predictor.predict_record(older)["mortality_probability"]
    assert p_old > p_young


def test_clinical_monotonicity_consciousness():
    """Clinically: drowsiness ('D') must yield higher mortality risk than fully alert ('F')."""
    predictor = StrokePredictor()
    base = dict(SYNTHETIC_RECORDS[0])

    alert = dict(base, consc="F")
    drowsy = dict(base, consc="D")

    p_alert = predictor.predict_record(alert)["mortality_probability"]
    p_drowsy = predictor.predict_record(drowsy)["mortality_probability"]
    assert p_drowsy > p_alert


def test_clinical_monotonicity_subtype():
    """Clinically: TACS (total anterior circulation) must yield higher risk than LACS (lacunar)."""
    predictor = StrokePredictor()
    base = dict(SYNTHETIC_RECORDS[0])

    lacs = dict(base, subtype="LACS")
    tacs = dict(base, subtype="TACS")

    p_lacs = predictor.predict_record(lacs)["mortality_probability"]
    p_tacs = predictor.predict_record(tacs)["mortality_probability"]
    assert p_tacs > p_lacs


def test_model_regression_snapshot():
    """Golden regression test: fixed synthetic patients must match the saved v1.1.0 baseline.

    The model is trained without class weighting, so predicted probabilities sit on the
    scale of the observed death rate (about 4.7%). The Youden threshold is about 0.049.
    """
    predictor = StrokePredictor()

    # Low risk synthetic patient: predicted risk about 0.005, below the threshold
    low = predictor.predict_record(SYNTHETIC_RECORDS[0])
    assert 0.001 <= low["mortality_probability"] <= 0.03
    assert low["high_risk_flag"] is False

    # High risk synthetic patient: predicted risk about 0.47, above the threshold
    high = predictor.predict_record(SYNTHETIC_RECORDS[1])
    assert 0.30 <= high["mortality_probability"] <= 0.65
    assert high["high_risk_flag"] is True


def test_probabilities_on_observed_risk_scale():
    """The typical synthetic patient must not receive a risk far above the 4.7% death rate.

    Guards against reintroducing class weighting, which shifts probabilities towards 50%.
    """
    predictor = StrokePredictor()
    mid_record = dict(SYNTHETIC_RECORDS[0], age=72.0, sbp=150.0, delay=14.0, subtype="PACS")
    assert predictor.predict_record(mid_record)["mortality_probability"] < 0.15
    assert predictor.threshold < 0.15

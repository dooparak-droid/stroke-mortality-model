"""API validation and integration tests.

Note: Authored/maintained by the project owner. Uses synthetic patient records only.
"""

from fastapi.testclient import TestClient
import pytest

from stroke_model.api import app

# Fixed synthetic record for testing (not a real patient)
SYNTHETIC_PATIENT = {
    "age": 72.0,
    "sbp": 145.0,
    "delay": 10.0,
    "gender": "F",
    "consc": "F",
    "subtype": "PACS",
    "treat1": "N",
    "treat2": "N",
    "wakesym": "N",
    "atrial": "N",
    "CT": "Y",
    "Infarc": "N",
    "hep24": "N",
    "asp3": "N",
    "symptom1": "Y",
    "symptom2": "Y",
    "symptom3": "N",
    "symptom4": "N",
    "symptom5": "N",
    "symptom6": "N",
    "symptom7": "N",
    "symptom8": "N"
}


def test_health_endpoint():
    """Verify /health returns 200 and model status."""
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "model_version" in data


def test_predict_valid_record():
    """Test prediction endpoint with valid synthetic clinical record."""
    with TestClient(app) as client:
        response = client.post("/predict", json=SYNTHETIC_PATIENT)
        assert response.status_code == 200
        payload = response.json()
        assert "mortality_probability" in payload
        assert 0.0 <= payload["mortality_probability"] <= 1.0
        assert isinstance(payload["high_risk_flag"], bool)
        assert "disclaimer" in payload


def test_predict_rejects_out_of_range_age():
    """Verify Pydantic schema rejects negative age or extreme values (>120)."""
    with TestClient(app) as client:
        invalid_record = dict(SYNTHETIC_PATIENT, age=-5.0)
        assert client.post("/predict", json=invalid_record).status_code == 422

        invalid_record2 = dict(SYNTHETIC_PATIENT, age=150.0)
        assert client.post("/predict", json=invalid_record2).status_code == 422


def test_predict_rejects_implausible_blood_pressure():
    """Verify Pydantic schema rejects SBP below 50 mmHg or above 300 mmHg."""
    with TestClient(app) as client:
        too_low = dict(SYNTHETIC_PATIENT, sbp=40.0)
        assert client.post("/predict", json=too_low).status_code == 422

        too_high = dict(SYNTHETIC_PATIENT, sbp=350.0)
        assert client.post("/predict", json=too_high).status_code == 422


def test_predict_rejects_implausible_delay():
    """Verify Pydantic schema rejects symptom onset delay > 100 hours."""
    with TestClient(app) as client:
        too_long = dict(SYNTHETIC_PATIENT, delay=150.0)
        assert client.post("/predict", json=too_long).status_code == 422


def test_predict_rejects_missing_field():
    """Verify Pydantic schema rejects payload when required predictor is omitted."""
    with TestClient(app) as client:
        incomplete_record = dict(SYNTHETIC_PATIENT)
        del incomplete_record["consc"]
        response = client.post("/predict", json=incomplete_record)
        assert response.status_code == 422


def test_predict_rejects_invalid_categorical_level():
    """Verify Pydantic schema rejects unknown category levels."""
    with TestClient(app) as client:
        invalid_record = dict(SYNTHETIC_PATIENT, consc="XYZ")
        response = client.post("/predict", json=invalid_record)
        assert response.status_code == 422


def test_api_version_matches_package_and_health_reports_model_version():
    """The API version is the package version, which is separate from the model version."""
    import json
    from importlib.metadata import version
    from pathlib import Path

    metadata = json.loads(
        (Path(__file__).resolve().parent.parent / "models" / "model_metadata.json").read_text()
    )
    assert app.version == version("stroke-model")
    with TestClient(app) as client:
        assert client.get("/health").json()["model_version"] == metadata["model_version"]

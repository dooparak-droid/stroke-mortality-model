"""Tests for the Streamlit front end. The API is replaced by fakes, so no service is needed."""

from pathlib import Path
from unittest.mock import patch

import requests
from streamlit.testing.v1 import AppTest

from stroke_model.features import ALL_PREDICTORS

APP = str(Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py")

HEALTH = {"status": "healthy", "model_version": "v2.0.0", "threshold": 0.0544}
PREDICTION = {
    "mortality_probability": 0.0213,
    "high_risk_flag": False,
    "threshold_applied": 0.0544,
    "model_version": "v2.0.0",
    "disclaimer": "Research demonstration only. Not for clinical use or medical decision-making.",
}


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = str(payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")


def fake_get_factory(health=HEALTH):
    def fake_get(url, timeout=None):
        if url.endswith("/health"):
            return FakeResponse(health)
        if url.endswith("/openapi.json"):
            return FakeResponse({"info": {"version": "0.1.0"}})
        raise AssertionError(f"unexpected GET {url}")
    return fake_get


def run_app(get=None, post=None):
    """Load the app with the fake service and return it together with the post mock."""
    with patch("requests.get", side_effect=get or fake_get_factory()), \
         patch("requests.post", side_effect=post or (lambda *a, **k: FakeResponse(PREDICTION))) as mock_post:
        at = AppTest.from_file(APP, default_timeout=30).run()
        return at, mock_post


def submit(at, mock_post, get=None):
    with patch("requests.get", side_effect=get or fake_get_factory()), \
         patch("requests.post", mock_post):
        at.button[0].click().run()
    return at


def checkbox(at, label):
    return next(c for c in at.checkbox if c.label == label)


def test_app_loads_and_shows_both_version_numbers():
    at, _ = run_app()
    assert not at.exception
    sidebar = " ".join(m.value for m in at.sidebar.markdown)
    assert "Model version v2.0.0" in sidebar
    assert "API version 0.1.0" in sidebar


def test_default_submission_sends_all_22_predictors_with_expected_codes():
    at, mock_post = run_app()
    submit(at, mock_post)

    payload = mock_post.call_args.kwargs["json"]
    assert set(payload) == set(ALL_PREDICTORS)          # matches the API schema exactly
    assert len(payload) == 22
    assert payload["age"] == 72.0 and payload["sbp"] == 150.0 and payload["delay"] == 14.0
    assert payload["subtype"] == "PACS" and payload["treat1"] == "N" and payload["treat2"] == "N"
    assert payload["CT"] == "Y" and payload["atrial"] == "N"
    assert payload["symptom1"] == "Y" and payload["symptom2"] == "Y" and payload["symptom3"] == "N"
    assert mock_post.call_args.args[0].endswith("/predict")

    values = {m.label: m.value for m in at.metric}
    assert values["Estimated risk of death by day 14"] == "2.1%"
    assert values["Screening flag"] == "Not flagged"
    assert values["Flag threshold"] == "5.4%"


def test_ticked_boxes_are_sent_as_Y():
    at, mock_post = run_app()
    checkbox(at, "Atrial fibrillation").check()
    checkbox(at, "Heparin in the 24 hours before randomisation").check()
    submit(at, mock_post)

    payload = mock_post.call_args.kwargs["json"]
    assert payload["atrial"] == "Y" and payload["hep24"] == "Y"


def test_high_risk_result_is_labelled():
    at, mock_post = run_app()
    high = dict(PREDICTION, mortality_probability=0.133, high_risk_flag=True)
    mock_post.side_effect = lambda *a, **k: FakeResponse(high)
    submit(at, mock_post)
    values = {m.label: m.value for m in at.metric}
    assert values["Screening flag"] == "High risk"
    assert values["Estimated risk of death by day 14"] == "13.3%"


def test_unreachable_service_shows_error_and_no_form():
    def down(url, timeout=None):
        raise requests.ConnectionError("refused")

    at, _ = run_app(get=down)
    assert not at.exception
    assert any("Cannot reach the service" in e.value for e in at.error)
    assert len(at.button) == 0


def test_a_different_service_is_rejected_with_a_message():
    at, _ = run_app(get=fake_get_factory(health={"status": "healthy", "version": "0.1.0"}))
    assert not at.exception
    assert any("does not look like the stroke prediction service" in e.value for e in at.error)


def test_validation_error_from_the_service_is_shown():
    at, mock_post = run_app()
    mock_post.side_effect = lambda *a, **k: FakeResponse({"detail": "age must be at most 120"}, 422)
    submit(at, mock_post)
    assert any("422" in e.value and "age must be at most 120" in e.value for e in at.error)
    assert len(at.metric) == 0


def test_age_outside_training_range_triggers_a_warning():
    at, mock_post = run_app()
    at.number_input[0].set_value(110)
    submit(at, mock_post)
    assert any("Outside the range of the training data" in w.value and "age 110" in w.value for w in at.warning)


def test_reference_figures_are_shown_only_for_the_matching_model_version():
    at, mock_post = run_app()
    submit(at, mock_post)
    assert any("For comparison" in m.value for m in at.markdown)

    at, mock_post = run_app()
    other = dict(PREDICTION, model_version="v9.9.9")
    mock_post.side_effect = lambda *a, **k: FakeResponse(other)
    submit(at, mock_post)
    assert not any("For comparison" in m.value for m in at.markdown)

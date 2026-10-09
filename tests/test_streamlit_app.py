"""Tests for the Streamlit app, in both of its modes. The API is replaced by fakes.

Default mode runs the real saved model inside the app. API mode talks to a fake service,
so no network or running service is needed.
"""

import os
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import requests
from streamlit.testing.v1 import AppTest

from stroke_model.features import ALL_PREDICTORS

APP_DIR = Path(__file__).resolve().parent.parent / "app"
APP = str(APP_DIR / "streamlit_app.py")
API_ENTRY = str(APP_DIR / "streamlit_app_api.py")
RENDER_URL = "https://stroke-mortality-model.onrender.com"

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


@contextmanager
def env(api_url=None):
    """Set or clear STROKE_API_URL for the duration of the block, then restore the original."""
    with patch.dict(os.environ):
        os.environ.pop("STROKE_API_URL", None)
        if api_url:
            os.environ["STROKE_API_URL"] = api_url
        yield


@contextmanager
def fake_service(get=None, post=None):
    with patch("requests.get", side_effect=get or fake_get_factory()), \
         patch("requests.post", side_effect=post or (lambda *a, **k: FakeResponse(PREDICTION))) as mock_post:
        yield mock_post


def checkbox(at, label):
    return next(c for c in at.checkbox if c.label == label)


def metrics(at):
    return {m.label: m.value for m in at.metric}


# --- API mode ----------------------------------------------------------------

API = "http://localhost:8010"


def test_api_mode_shows_proof_of_concept_notice_and_both_version_numbers():
    with env(API), fake_service():
        at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    sidebar = " ".join(m.value for m in at.sidebar.markdown)
    assert "Mode: calls the API" in sidebar
    assert "Model version v2.0.0" in sidebar
    assert "API version 0.1.0" in sidebar
    assert any("Proof of concept" in i.value for i in at.info)


def test_api_mode_sends_all_22_predictors_with_expected_codes():
    with env(API), fake_service() as mock_post:
        at = AppTest.from_file(APP, default_timeout=30).run()
        at.button[0].click().run()

    payload = mock_post.call_args.kwargs["json"]
    assert set(payload) == set(ALL_PREDICTORS)          # matches the API schema exactly
    assert len(payload) == 22
    assert payload["age"] == 72.0 and payload["sbp"] == 150.0 and payload["delay"] == 14.0
    assert payload["subtype"] == "PACS" and payload["treat1"] == "N" and payload["treat2"] == "N"
    assert payload["CT"] == "Y" and payload["atrial"] == "N"
    assert payload["symptom1"] == "Y" and payload["symptom2"] == "Y" and payload["symptom3"] == "N"
    assert mock_post.call_args.args[0] == f"{API}/predict"

    values = metrics(at)
    assert values["Estimated risk of death by day 14"] == "2.1%"
    assert values["Screening flag"] == "Not flagged"
    assert values["Flag threshold"] == "5.4%"


def test_api_mode_ticked_boxes_are_sent_as_Y():
    with env(API), fake_service() as mock_post:
        at = AppTest.from_file(APP, default_timeout=30).run()
        checkbox(at, "Atrial fibrillation").check()
        checkbox(at, "Heparin in the 24 hours before randomisation").check()
        at.button[0].click().run()
    payload = mock_post.call_args.kwargs["json"]
    assert payload["atrial"] == "Y" and payload["hep24"] == "Y"


def test_api_mode_high_risk_result_is_labelled():
    high = dict(PREDICTION, mortality_probability=0.133, high_risk_flag=True)
    with env(API), fake_service(post=lambda *a, **k: FakeResponse(high)):
        at = AppTest.from_file(APP, default_timeout=30).run()
        at.button[0].click().run()
    values = metrics(at)
    assert values["Screening flag"] == "High risk"
    assert values["Estimated risk of death by day 14"] == "13.3%"


def test_api_mode_unreachable_service_shows_error_and_no_form():
    def down(url, timeout=None):
        raise requests.ConnectionError("refused")

    with env(API), fake_service(get=down):
        at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert any("Cannot reach the service" in e.value for e in at.error)
    assert len(at.button) == 0


def test_api_mode_a_different_service_is_rejected_with_a_message():
    other = fake_get_factory(health={"status": "healthy", "version": "0.1.0"})
    with env(API), fake_service(get=other):
        at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert any("does not look like the stroke prediction service" in e.value for e in at.error)


def test_api_mode_validation_error_from_the_service_is_shown():
    bad = lambda *a, **k: FakeResponse({"detail": "age must be at most 120"}, 422)
    with env(API), fake_service(post=bad):
        at = AppTest.from_file(APP, default_timeout=30).run()
        at.button[0].click().run()
    assert any("422" in e.value and "age must be at most 120" in e.value for e in at.error)
    assert len(at.metric) == 0


def test_api_mode_reference_figures_only_for_the_matching_model_version():
    with env(API), fake_service():
        at = AppTest.from_file(APP, default_timeout=30).run()
        at.button[0].click().run()
    assert any("For comparison" in m.value for m in at.markdown)

    other = dict(PREDICTION, model_version="v9.9.9")
    with env(API), fake_service(post=lambda *a, **k: FakeResponse(other)):
        at = AppTest.from_file(APP, default_timeout=30).run()
        at.button[0].click().run()
    assert not any("For comparison" in m.value for m in at.markdown)


# --- Default mode: the model runs inside the app ------------------------------

def never_called(*args, **kwargs):
    raise AssertionError("the default mode must not make network requests")


def test_default_mode_runs_the_real_model_without_any_network_call():
    with env(), fake_service(get=never_called, post=never_called):
        at = AppTest.from_file(APP, default_timeout=60).run()
        assert not at.exception
        sidebar = " ".join(m.value for m in at.sidebar.markdown)
        assert "Mode: model runs inside this app" in sidebar
        assert "Model version v2.0.0" in sidebar
        assert not any("Proof of concept" in i.value for i in at.info)

        at.button[0].click().run()
        assert not at.exception

    values = metrics(at)
    assert values["Estimated risk of death by day 14"] == "2.1%"      # the README example patient
    assert values["Screening flag"] == "Not flagged"
    assert values["Flag threshold"] == "5.4%"


def test_default_mode_flags_a_high_risk_patient():
    with env(), fake_service(get=never_called, post=never_called):
        at = AppTest.from_file(APP, default_timeout=60).run()
        at.number_input[0].set_value(95)                      # age
        at.radio[1].set_value("D")                            # drowsy
        at.selectbox[0].set_value("TACS")                     # total anterior circulation stroke
        at.button[0].click().run()
    assert not at.exception
    assert metrics(at)["Screening flag"] == "High risk"


def test_default_mode_warns_for_age_outside_the_training_range():
    with env(), fake_service(get=never_called, post=never_called):
        at = AppTest.from_file(APP, default_timeout=60).run()
        at.number_input[0].set_value(110)
        at.button[0].click().run()
    assert any("Outside the range of the training data" in w.value and "age 110" in w.value for w in at.warning)


# --- Proof-of-concept entry point ----------------------------------------------

def test_api_entry_point_calls_the_deployed_api_by_default():
    with env(), fake_service() as mock_post:
        at = AppTest.from_file(API_ENTRY, default_timeout=30).run()
        assert not at.exception
        at.button[0].click().run()
    assert mock_post.call_args.args[0] == f"{RENDER_URL}/predict"
    assert any("Proof of concept" in i.value for i in at.info)


def test_api_entry_point_respects_an_address_that_is_already_set():
    with env(API), fake_service() as mock_post:
        at = AppTest.from_file(API_ENTRY, default_timeout=30).run()
        at.button[0].click().run()
    assert mock_post.call_args.args[0] == f"{API}/predict"

"""streamlit_app.py

Browser front end for the stroke mortality prediction service. It sends the form values
to the FastAPI service (default http://localhost:8000) and shows the predicted risk of
death by day 14. Start the service first, then run:

    streamlit run app/streamlit_app.py

To use the deployed service instead, set STROKE_API_URL to its address.
"""

import json
import os

import requests
import streamlit as st

API_URL = os.getenv("STROKE_API_URL", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT_SECONDS = 90  # a service on free hosting can take about a minute to wake

# Ranges of the training data (International Stroke Trial subset used for model v2.0.0)
TRAINING_RANGES = {
    "age": (16, 98, "years"),
    "sbp": (70, 295, "mmHg"),
    "delay": (1, 48, "hours"),
}

# Reference figures that describe model v2.0.0. They are shown only when the service reports
# that version, so they cannot go stale after the model is retrained.
REFERENCE_MODEL_VERSION = "v2.0.0"
TRAINING_EVENT_RATE = 0.051  # 667 deaths among 13,063 training-file patients
HOLDOUT_FLAGGED_SHARE = 0.281
HOLDOUT_FLAGGED_DIED = 0.122
HOLDOUT_SENSITIVITY = 0.671

YES_NO = ["No", "Yes"]
DEFICITS = [
    ("symptom1", "Face deficit"),
    ("symptom2", "Arm or hand deficit"),
    ("symptom3", "Leg or foot deficit"),
    ("symptom4", "Dysphasia"),
    ("symptom5", "Hemianopia"),
    ("symptom6", "Visuospatial disorder"),
    ("symptom7", "Brainstem or cerebellar signs"),
    ("symptom8", "Other deficit"),
]
SUBTYPES = {
    "LACS": "Lacunar (LACS)",
    "PACS": "Partial anterior circulation (PACS)",
    "POCS": "Posterior circulation (POCS)",
    "TACS": "Total anterior circulation (TACS)",
    "OTH": "Other",
}

st.set_page_config(page_title="Stroke 14-day mortality risk", layout="wide")
st.title("Stroke 14-day mortality risk")
st.warning(
    "Research demonstration only. Not for clinical use or medical decision-making. "
    "The model was trained on trial data from the 1990s."
)


def get_json(path, timeout):
    response = requests.get(f"{API_URL}{path}", timeout=timeout)
    response.raise_for_status()
    return response.json()


if "health" not in st.session_state:
    try:
        with st.spinner("Contacting the service. A service on free hosting can take about a minute to wake."):
            st.session_state["health"] = get_json("/health", REQUEST_TIMEOUT_SECONDS)
        try:
            st.session_state["api_version"] = get_json("/openapi.json", 10)["info"]["version"]
        except (requests.RequestException, KeyError, ValueError):
            st.session_state["api_version"] = None
    except (requests.RequestException, ValueError):
        st.error(f"Cannot reach the service at {API_URL}. Start it first, then reload this page.")
        st.stop()

health = st.session_state["health"]
if not isinstance(health, dict) or not {"model_version", "threshold"} <= set(health):
    st.session_state.pop("health", None)
    st.error(f"The service at {API_URL} does not look like the stroke prediction service.")
    st.stop()

with st.sidebar:
    st.subheader("Service")
    st.write(f"Model version {health['model_version']}")
    if st.session_state.get("api_version"):
        st.write(f"API version {st.session_state['api_version']}")
    st.write(f"High-risk threshold {health['threshold']:.1%}")
    st.caption(
        "The model version identifies the trained model. The API version identifies the software "
        "that serves it."
    )

st.write(
    "Enter the patient's details as recorded at randomisation in the International Stroke Trial. "
    "The model estimates the risk of death within 14 days."
)

with st.form("patient"):
    st.subheader("Patient")
    c1, c2, c3 = st.columns(3)
    age = c1.number_input("Age (years)", min_value=0, max_value=120, value=72)
    gender = c2.radio("Sex", ["F", "M"], format_func={"F": "Female", "M": "Male"}.get, horizontal=True)
    consc = c3.radio(
        "Conscious state", ["F", "D"],
        format_func={"F": "Fully alert", "D": "Drowsy"}.get, horizontal=True,
    )

    st.subheader("Presentation")
    c1, c2, c3 = st.columns(3)
    sbp = c1.number_input("Systolic blood pressure (mmHg)", min_value=50, max_value=300, value=150)
    delay = c2.number_input("Hours from stroke onset to randomisation", min_value=0, max_value=100, value=14)
    subtype = c3.selectbox("Stroke subtype", list(SUBTYPES), index=1, format_func=SUBTYPES.get)
    c1, c2, c3 = st.columns(3)
    wakesym = c1.checkbox("Symptoms noted on waking")
    atrial = c2.checkbox("Atrial fibrillation")
    ct = c3.checkbox("CT scan before randomisation", value=True)
    infarc = c1.checkbox("Infarct visible on CT")

    st.subheader("Neurological deficits at randomisation")
    deficit_columns = st.columns(4)
    deficits = {}
    for i, (key, label) in enumerate(DEFICITS):
        deficits[key] = deficit_columns[i % 4].checkbox(label, value=key in ("symptom1", "symptom2"))

    st.subheader("Treatment")
    c1, c2, c3, c4 = st.columns(4)
    hep24 = c1.checkbox("Heparin in the 24 hours before randomisation")
    asp3 = c2.checkbox("Aspirin in the 3 days before randomisation")
    treat1 = c3.selectbox(
        "Trial heparin allocation", ["N", "L", "M"],
        format_func={"N": "None", "L": "Low dose", "M": "Medium dose"}.get,
    )
    treat2 = c4.selectbox(
        "Trial aspirin allocation", ["N", "Y"],
        format_func={"N": "No aspirin", "Y": "Aspirin"}.get,
    )

    submitted = st.form_submit_button("Estimate risk", type="primary")


def yn(flag):
    return "Y" if flag else "N"


if submitted:
    payload = {
        "age": float(age),
        "sbp": float(sbp),
        "delay": float(delay),
        "gender": gender,
        "consc": consc,
        "subtype": subtype,
        "treat1": treat1,
        "treat2": treat2,
        "wakesym": yn(wakesym),
        "atrial": yn(atrial),
        "CT": yn(ct),
        "Infarc": yn(infarc),
        "hep24": yn(hep24),
        "asp3": yn(asp3),
        **{key: yn(value) for key, value in deficits.items()},
    }
    with st.spinner("Estimating risk."):
        try:
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
        except requests.RequestException as e:
            st.session_state.pop("result", None)
            st.error(f"Request failed: {e}")
        else:
            if response.status_code == 200:
                st.session_state["result"] = {"input": payload, "output": response.json()}
            else:
                st.session_state.pop("result", None)
                try:
                    detail = response.json().get("detail", response.text)
                except ValueError:
                    detail = response.text
                st.error(f"Service returned {response.status_code}: {detail}")

result = st.session_state.get("result")
if result:
    output, sent = result["output"], result["input"]
    risk = output["mortality_probability"]
    st.divider()
    st.subheader("Result")

    outside = [
        f"{name} {sent[name]:g} {unit} (training data {low} to {high})"
        for name, (low, high, unit) in TRAINING_RANGES.items()
        if not low <= sent[name] <= high
    ]
    if outside:
        st.warning(
            "Outside the range of the training data: " + "; ".join(outside)
            + ". The estimate is an extrapolation."
        )

    m1, m2, m3 = st.columns(3)
    m1.metric("Estimated risk of death by day 14", f"{risk:.1%}")
    m2.metric("Screening flag", "High risk" if output["high_risk_flag"] else "Not flagged")
    m3.metric("Flag threshold", f"{output['threshold_applied']:.1%}")

    if output["model_version"] == REFERENCE_MODEL_VERSION:
        st.write(
            f"For comparison, {TRAINING_EVENT_RATE:.1%} of the patients in the training data died within "
            f"14 days. This estimate is {risk / TRAINING_EVENT_RATE:.1f} times that rate."
        )
        with st.expander("How to read the flag"):
            st.write(
                f"The flag is a screening flag. The threshold ({output['threshold_applied']:.1%}) is low "
                "because deaths are uncommon, and it was chosen to balance sensitivity and specificity. "
                f"On held-out patients it flagged {HOLDOUT_FLAGGED_SHARE:.0%} of everyone and caught "
                f"{HOLDOUT_SENSITIVITY:.0%} of the deaths, but only {HOLDOUT_FLAGGED_DIED:.0%} of the "
                "patients it flagged died. A patient who is not flagged is not risk free."
            )

    st.caption(output["disclaimer"])
    st.download_button(
        "Download result (JSON)",
        json.dumps({"input": sent, "output": output}, indent=2),
        "stroke_risk_result.json",
        "application/json",
    )

st.divider()
st.caption(
    "Model trained on a subset of the International Stroke Trial database (Sandercock, Niewada and "
    "Czlonkowska, University of Edinburgh, ODC-By v1.0, https://doi.org/10.7488/ds/104). "
    "Source code and documentation: https://github.com/dooparak-droid/stroke-mortality-model"
)

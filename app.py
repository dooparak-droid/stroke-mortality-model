"""Gradio web application for interactive clinical stroke mortality prediction on Hugging Face Spaces."""

import gradio as gr
from stroke_model.predict import StrokePredictor

# Initialize predictor
predictor = StrokePredictor()


def predict_mortality_ui(
    age: float,
    sbp: float,
    delay: float,
    gender: str,
    consc: str,
    subtype: str,
    treat1: str,
    treat2: str,
    wakesym: str,
    atrial: str,
    CT: str,
    Infarc: str,
    hep24: str,
    asp3: str,
    symptom1: str,
    symptom2: str,
    symptom3: str,
    symptom4: str,
    symptom5: str,
    symptom6: str,
    symptom7: str,
    symptom8: str
):
    """Bridge Gradio inputs to StrokePredictor and format output for display."""
    record = {
        "age": age,
        "sbp": sbp,
        "delay": delay,
        "gender": gender,
        "consc": consc,
        "subtype": subtype,
        "treat1": treat1,
        "treat2": treat2,
        "wakesym": wakesym,
        "atrial": atrial,
        "CT": CT,
        "Infarc": Infarc,
        "hep24": hep24,
        "asp3": asp3,
        "symptom1": symptom1,
        "symptom2": symptom2,
        "symptom3": symptom3,
        "symptom4": symptom4,
        "symptom5": symptom5,
        "symptom6": symptom6,
        "symptom7": symptom7,
        "symptom8": symptom8
    }

    result = predictor.predict_record(record)
    prob = result["mortality_probability"]
    is_high_risk = result["high_risk_flag"]
    threshold = result["threshold_applied"]

    risk_label = "HIGH RISK" if is_high_risk else "LOW / MODERATE RISK"
    prob_percentage = f"{prob * 100:.1f}%"

    summary_md = f"""
    ### 🏥 Prediction Results

    - **Estimated 14-Day Mortality Risk:** `{prob_percentage}` (Probability: `{prob:.4f}`)
    - **Classification Status:** **{risk_label}**
    - **Decision Threshold Applied:** `{threshold:.4f}` (Youden's J threshold)
    - **Model Version:** `{result['model_version']}`

    > ⚠️ **Clinical Disclaimer:** {result['disclaimer']}
    """

    return summary_md, {"High Risk": prob, "Survival / Low Risk": 1.0 - prob}


# Construct Gradio Interface
demo = gr.Interface(
    fn=predict_mortality_ui,
    inputs=[
        gr.Slider(minimum=18, maximum=100, value=72, step=1, label="Age (years)"),
        gr.Slider(minimum=60, maximum=260, value=150, step=1, label="Systolic Blood Pressure (mmHg)"),
        gr.Slider(minimum=0, maximum=96, value=12, step=1, label="Onset to Admission Delay (hours)"),
        gr.Radio(choices=["F", "M"], value="F", label="Gender ('F' = Female, 'M' = Male)"),
        gr.Radio(choices=["F", "D"], value="F", label="Consciousness Level ('F' = Alert, 'D' = Drowsy)"),
        gr.Dropdown(choices=["LACS", "OTH", "PACS", "POCS", "TACS"], value="PACS", label="OCSP Stroke Subtype"),
        gr.Radio(choices=["L", "M", "N"], value="N", label="Treatment Code 1 ('L', 'M', 'N')"),
        gr.Radio(choices=["N", "Y"], value="N", label="Treatment Code 2 ('N' / 'Y')"),
        gr.Radio(choices=["N", "Y"], value="N", label="Wake-up Stroke Symptoms ('N' / 'Y')"),
        gr.Radio(choices=["N", "Y"], value="N", label="Atrial Fibrillation ('N' / 'Y')"),
        gr.Radio(choices=["N", "Y"], value="Y", label="CT Scan Performed ('N' / 'Y')"),
        gr.Radio(choices=["N", "Y"], value="N", label="Infarction Confirmed ('N' / 'Y')"),
        gr.Radio(choices=["N", "Y"], value="N", label="Heparin within 24 Hours ('N' / 'Y')"),
        gr.Radio(choices=["N", "Y"], value="N", label="Aspirin within 3 Days ('N' / 'Y')"),
        gr.Radio(choices=["N", "Y"], value="Y", label="Neurological Deficit Symptom 1"),
        gr.Radio(choices=["N", "Y"], value="Y", label="Neurological Deficit Symptom 2"),
        gr.Radio(choices=["N", "Y"], value="N", label="Neurological Deficit Symptom 3"),
        gr.Radio(choices=["N", "Y"], value="N", label="Neurological Deficit Symptom 4"),
        gr.Radio(choices=["N", "Y"], value="N", label="Neurological Deficit Symptom 5"),
        gr.Radio(choices=["N", "Y"], value="N", label="Neurological Deficit Symptom 6"),
        gr.Radio(choices=["N", "Y"], value="N", label="Neurological Deficit Symptom 7"),
        gr.Radio(choices=["N", "Y"], value="N", label="Neurological Deficit Symptom 8")
    ],
    outputs=[
        gr.Markdown(label="Clinical Summary"),
        gr.Label(label="Risk Probability Distribution")
    ],
    title="Stroke 14-Day Mortality Risk Predictor",
    description=(
        "Interactive machine learning demonstration for early post-stroke 14-day mortality risk assessment. "
        "Ported from R glmnet analysis to scikit-learn Ridge regularised logistic regression. "
        "Research demonstration only — not for clinical decision-making."
    ),
    flagging_mode="never"
)

if __name__ == "__main__":
    demo.launch()

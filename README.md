# Stroke Mortality Prediction Service

> **Research demonstration only. Not for clinical use or medical decision-making.**

This project ports, packages, and serves a machine learning model predicting 14-day mortality following acute stroke. It builds upon earlier exploratory analysis conducted in R (available at [Evaluating-survival-probability-after-a-stroke](https://github.com/dooparak-droid/Evaluating-survival-probability-after-a-stroke)), re-implementing the complete lifecycle in Python with an emphasis on rigorous validation, model calibration, automated testing, containerised deployment, and drift monitoring.

---

## Clinical and Methodological Context

In acute stroke care, early mortality risk assessment can assist in triage, resource planning, and identifying patients at elevated risk of deterioration. 

A primary methodological challenge in this cohort is the severe **class imbalance**, with an observed event rate of **4.7%**. Standard evaluation approaches that focus purely on accuracy or discrimination (such as the Area Under the ROC Curve) can be misleading:
* A model can achieve an AUC of 0.80 while still predicting risks that are systematically too high or too low.
* For rare adverse events, **calibration** (assessed via calibration curves and Brier score) is vital to ensure that a predicted probability of 10% genuinely corresponds to roughly 10 observed deaths per 100 similar patients.
* The classification threshold must be chosen deliberately. Here, **Youden's J statistic** is optimised strictly on cross-validation folds to balance sensitivity and specificity without data leakage into the test set.

---

## Project Structure

```
stroke-mortality-model/
├── .github/workflows/ci.yml    # Continuous integration with automated testing
├── Dockerfile                  # Container definition for reproducible deployment
├── monitoring/                 # Reference distributions and drift detection
├── models/                     # Serialised model pipelines and metadata
├── reports/                    # Calibration curves, ROC plots, and evaluation tables
├── src/stroke_model/
│   ├── __init__.py
│   ├── api.py                  # FastAPI service with validated schemas
│   ├── data.py                 # Data loading and stratified splitting
│   ├── features.py             # Feature definitions and preprocessing pipeline
│   ├── train.py                # Hyperparameter tuning and model training
│   ├── evaluate.py             # Calibration, Brier score, and discrimination metrics
│   └── predict.py              # Inference interface
├── tests/                      # Unit, schema validation, and regression tests
├── pyproject.toml              # Build system and package configuration
├── requirements.txt            # Pinned dependencies
└── README.md
```

---

## Local Setup

### 1. Environment and Dependencies

Ensure Python 3.10 or higher is installed. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e ".[dev]"
```

### 2. Training the Model

Place `assignment2026.csv` in the root workspace (this file is excluded from version control via `.gitignore`). Run the training script:

```bash
python -m stroke_model.train
```

This performs:
1. Stratified 75/25 train-test partitioning.
2. 10-fold cross-validation with inverse-frequency class weighting.
3. Optimal regularisation tuning for Ridge logistic regression.
4. Out-of-fold threshold optimisation using Youden's J statistic.
5. Export of the versioned model pipeline artifact to `models/`.

### 3. Running the Test Suite

```bash
pytest -v tests/
```

### 4. Running the API Locally

```bash
uvicorn stroke_model.api:app --host 0.0.0.0 --port 8000 --reload
```

Interactive documentation is available at `http://localhost:8000/docs`.

---

## Running with Docker

Build and run the containerised application:

```bash
docker build -t stroke-mortality-model .
docker run -p 7860:7860 stroke-mortality-model
```

Access the health endpoint:
```bash
curl http://localhost:7860/health
```

---

## API Specification

### Health Check: `GET /health`
Returns service status and the active model version.

### Prediction: `POST /predict`
Validates 22 patient predictors and returns the estimated mortality probability alongside the classification decision based on the pre-specified Youden threshold.

Example request using `curl`:
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "age": 72.0,
       "sbp": 150.0,
       "delay": 14.0,
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
     }'
```

Example response:
```json
{
  "mortality_probability": 0.4128,
  "high_risk_flag": false,
  "threshold_applied": 0.5498,
  "model_version": "v1.0.0",
  "disclaimer": "Research demonstration only. Not for clinical use or medical decision-making."
}
```

---

## Results and Comparison with R Baseline

Both pipelines utilise inverse-frequency class weighting (approximately 20.3x for deaths) to address the 4.7% event rate. A key methodological distinction is threshold selection: the R baseline selected Youden's threshold on the holdout test set, whereas the Python implementation optimised the threshold strictly across out-of-fold cross-validation folds before applying it to the untouched holdout test partition.

| Metric | R Baseline (glmnet) | Python Implementation (scikit-learn) | Notes |
|---|---|---|---|
| Model Architecture | Ridge Logistic Regression (L2) | Ridge Logistic Regression (L2) | Standardised predictors |
| Cross-validation AUC | 0.7813 | 0.7652 | 10-fold stratified CV |
| Test Partition AUC | Not reported separately | 0.8029 | 25% holdout (3,266 patients) |
| Probability Calibration (Brier Score) | Not evaluated | 0.1891 | Mean squared probability error |
| Youden Decision Threshold | 0.50 (nominal) / 0.55 | 0.5498 | Tuned out-of-fold in Python |
| Test Sensitivity | 69.1% | 65.4% | True positive rate |
| Test Specificity | 75.3% | 79.9% | True negative rate |
| Test F1-Score | 0.2016 | 0.2275 | Harmonic mean of precision/recall |

The calibration curve is stored at `reports/calibration_plot.png`. Due to inverse-frequency weighting during training, predicted probabilities represent weighted risk scores; post-hoc calibration can be applied if absolute population incidence probabilities are required.

---

## Limitations and Governance

1. **Synthetic / Benchmark Data:** The dataset is derived or anonymised without individual patient identifiers. It does not represent validated clinical records from any specific healthcare provider.
2. **No External Validation:** The model has only undergone internal cross-validation and split-sample holdout testing. Generalisability across different clinical settings remains unverified.
3. **Research Demonstration:** This repository is an educational software engineering and machine learning lifecycle demonstration. It must not be deployed in real clinical care or used to influence medical decisions.

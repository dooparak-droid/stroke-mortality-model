# Stroke Mortality Prediction Service

> **Research demonstration only. Not for clinical use or medical decision-making.**

This project ports, packages, and serves a machine learning model predicting 14-day mortality following acute stroke. It builds upon earlier exploratory analysis conducted in R (available at [Evaluating-survival-probability-after-a-stroke](https://github.com/dooparak-droid/Evaluating-survival-probability-after-a-stroke)), re-implementing the complete lifecycle in Python with an emphasis on rigorous validation, model calibration, automated testing, containerised deployment, and drift monitoring.

---

## Live Demo

The service is deployed on Render's free tier at [stroke-mortality-model.onrender.com](https://stroke-mortality-model.onrender.com). Opening the link redirects to the interactive API documentation at `/docs`, where the `POST /predict` endpoint can be tried directly in the browser.

Two limitations of the free tier apply. The service sleeps after 15 minutes without traffic, so the first request after a quiet period can take about a minute. The filesystem is also reset on every restart, so prediction logs written by the deployed service are not kept. The drift check in `monitoring/` is intended to be run on logs from a local run.

---

## Clinical and Methodological Context

In acute stroke care, early mortality risk assessment can assist in triage, resource planning, and identifying patients at elevated risk of deterioration. 

A primary methodological challenge in this cohort is the severe **class imbalance**, with an observed event rate of **4.7%**. Standard evaluation approaches that focus purely on accuracy or discrimination (such as the Area Under the ROC Curve) can be misleading:
* A model can achieve an AUC of 0.80 while still predicting risks that are systematically too high or too low.
* For rare adverse events, **calibration** (assessed via calibration curves and Brier score) is vital to ensure that a predicted probability of 10% genuinely corresponds to roughly 10 observed deaths per 100 similar patients.
* The classification threshold must be chosen deliberately. Here, **Youden's J statistic** is optimised strictly on cross-validation folds to balance sensitivity and specificity without data leakage into the test set. Because the event rate is low, the resulting threshold on predicted risk is also low (about 0.049), and the high-risk flag is a screening flag and not a confident prediction of death.

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
2. 10-fold cross-validation to tune the ridge penalty, with no class weighting.
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
  "mortality_probability": 0.0187,
  "high_risk_flag": false,
  "threshold_applied": 0.0494,
  "model_version": "v1.1.0",
  "disclaimer": "Research demonstration only. Not for clinical use or medical decision-making."
}
```

---

## Results and Comparison with R Baseline

The R baseline trained its ridge model with inverse-frequency class weighting (deaths counted about 20 times as heavily as survivors). The Python model is trained without class weighting, because the weighting raised every predicted probability towards 50% and left it unusable as a risk estimate. The low event rate is handled instead by the classification threshold. Threshold selection also differs. The R baseline selected Youden's threshold on the holdout test set, whereas the Python implementation optimised the threshold across out-of-fold cross-validation folds before applying it to the untouched holdout test partition.

| Metric | R Baseline (glmnet, weighted) | Python Implementation (scikit-learn, unweighted) | Notes |
|---|---|---|---|
| Model Architecture | Ridge Logistic Regression (L2) | Ridge Logistic Regression (L2) | Standardised predictors |
| Cross-validation AUC | 0.7813 | 0.7657 | 10-fold stratified CV on the training partition |
| Test Partition AUC | Not reported separately | 0.8048 | 25% holdout (3,266 patients) |
| Brier Score | 0.188 | 0.0412 | Mean squared error of predicted risk. Lower is better. The R value was computed afterwards from the saved R predictions, not reported in the original analysis |
| Mean Predicted Risk | 39.8% | 4.7% | Observed death rate in the holdout is 4.7% |
| Youden Decision Threshold | 0.50 (nominal) / 0.55 | 0.0494 | Cutoff on predicted risk. Tuned out-of-fold in Python |
| Test Sensitivity | 69.1% | 73.2% | Share of deaths flagged high risk |
| Test Specificity | 75.3% | 75.0% | Share of survivors not flagged |
| Test Precision | Not reported | 12.6% | Share of flagged patients who died |
| Test F1-Score | 0.2016 | 0.2146 | Harmonic mean of precision and sensitivity |

For reference, a model that gave every patient the same risk of 4.7% would have a Brier score of about 0.045. The R model's Brier score is far above this because of the weighting. An earlier Python run with the same class weighting as R gave a test AUC of 0.8029 and a Brier score of 0.1891, so removing the weighting did not change how well patients are ranked.

At the Python threshold of 0.0494, about 27% of holdout patients are flagged high risk, and about 13% of flagged patients died. The flag is therefore a screening flag.

The calibration curve is stored at `reports/calibration_plot.png`. It sorts the holdout patients into ten equal groups of about 330 by predicted risk and compares the mean predicted risk in each group with the proportion who died. In the highest-risk group the two agree closely (predicted 0.18, observed 0.19). In the groups with predicted risk below about 0.035 the model slightly overestimates risk, with each of these groups containing fewer than 10 deaths. In two middle groups it underestimates (predicted 0.053 and 0.080, observed 0.083 and 0.089), and these groups contain about 27 and 29 deaths. Differences of this size could partly be chance. No recalibration was applied.

---

## Limitations and Governance

1. **Synthetic / Benchmark Data:** The dataset is derived or anonymised without individual patient identifiers. It does not represent validated clinical records from any specific healthcare provider.
2. **No External Validation:** The model has only undergone internal cross-validation and split-sample holdout testing. Generalisability across different clinical settings remains unverified.
3. **Research Demonstration:** This repository is an educational software engineering and machine learning lifecycle demonstration. It must not be deployed in real clinical care or used to influence medical decisions.

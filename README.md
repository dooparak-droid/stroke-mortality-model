# Stroke Mortality Prediction Service

> **Research demonstration only. Not for clinical use or medical decision-making.**

This project ports, packages, and serves a machine learning model predicting 14-day mortality following acute stroke. It builds upon earlier exploratory analysis conducted in R (available at [Evaluating-survival-probability-after-a-stroke](https://github.com/dooparak-droid/Evaluating-survival-probability-after-a-stroke)), re-implementing the complete lifecycle in Python with an emphasis on rigorous validation, model calibration, automated testing, containerised deployment, and drift monitoring.

---

## Live Demo

The service is deployed on Render's free tier at [stroke-mortality-model.onrender.com](https://stroke-mortality-model.onrender.com). Opening the link redirects to the interactive API documentation at `/docs`, where the `POST /predict` endpoint can be tried directly in the browser.

Two limitations of the free tier apply. The service sleeps after 15 minutes without traffic, so the first request after a quiet period can take about a minute. The service does not log requests. The drift check is run separately on a CSV of input records (see Drift Monitoring below).

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
│   ├── predict.py              # Inference interface
│   └── reference.py            # Training reference bins and Population Stability Index
├── tests/                      # Unit, schema validation, regression, and drift tests
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

Place `stroke_dataset.csv` in the root workspace (this file is excluded from version control via `.gitignore`). Run the training script:

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

## Drift Monitoring

The drift check compares a batch of incoming records with the training data and reports how much each predictor's distribution has changed. It uses the **Population Stability Index (PSI)**, a single number per predictor that is 0 when the new records are distributed exactly like the training records and grows as they diverge. For a numeric predictor such as age, the training patients are sorted into ten bins holding about 10% of patients each. The share of new patients falling in each bin is then compared with the training share. For each bin, the difference between the two shares is multiplied by the natural logarithm of their ratio, and the ten results are added. For a categorical predictor, each category is a bin.

The conventional reading of PSI is below 0.1 for stable, 0.1 to 0.2 for a moderate shift to monitor, and 0.2 or above for a shift that needs investigation. The report also gives the shift in the mean of each numeric predictor in training standard deviations and the largest change in any category share. These two simpler checks can miss a change in spread. If every new patient had the training mean age, the mean would not move at all, but PSI would flag the change.

The training step stores the bin edges and the training share in each bin for age, systolic blood pressure and delay in `monitoring/reference_stats.json`, together with category shares for the other predictors. No individual patient records are stored.

```bash
python monitoring/drift_check.py path/to/inputs.csv
```

The CSV needs columns named as in the API schema, and any predictor it lacks is skipped. Without a file, the script uses a synthetic sample of 500 records from an older cohort with higher blood pressure. On that sample the mean-shift checks report no alert for age, systolic blood pressure or delay, while PSI reports alerts for all three (0.34, 0.29 and 1.76).

Limitations:
* The deployed service does not log requests, so the CSV has to be assembled separately.
* PSI is noisy for small batches, and the report adds a note when there are fewer than 100 records.
* PSI measures change in the inputs only. It does not show whether the model's predictions have become less accurate, which would need observed outcomes.

---

## API Specification

### Version numbers

The service reports two version numbers that mean different things. The API version (shown on the `/docs` page) is the version of the software, set by `version` in `pyproject.toml` and currently 0.1.0. The model version (`model_version` in the `/health` and `/predict` responses) is the version of the trained model artefact in `models/`, currently v1.1.0. The API version changes when the code changes, and the model version changes when the model is retrained.

### Health Check: `GET /health`
Returns service status, the active model version and the classification threshold.

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

The R figures are taken from the output files saved by the R analysis on 16 February 2026. The R script was edited afterwards, and two of the saved files differ slightly from each other (for example test AUC 0.7806 in one and 0.7813 in the other), so the R values are approximate. The two analyses also use different splitting functions, so their test partitions are not identical. The R baseline trained its ridge model with inverse-frequency class weighting (deaths counted about 20 times as heavily as survivors). The Python model is trained without class weighting, because the weighting raised every predicted probability towards 50% and left it unusable as a risk estimate. The low event rate is handled instead by the classification threshold. Threshold selection also differs. The R baseline selected Youden's threshold on the holdout test set, whereas the Python implementation optimised the threshold across out-of-fold cross-validation folds before applying it to the untouched holdout test partition.

| Metric | R Baseline (glmnet, weighted) | Python Implementation (scikit-learn, unweighted) | Notes |
|---|---|---|---|
| Model Architecture | Ridge Logistic Regression (L2) | Ridge Logistic Regression (L2) | Standardised predictors |
| Cross-validation AUC | 0.7718 | 0.7657 | 10-fold CV on the training partition |
| Test Partition AUC | 0.7813 | 0.8048 (95% CI 0.7695 to 0.8385) | 25% holdout (about 3,265 patients in each analysis). The interval is a bootstrap interval |
| Brier Score | 0.188 | 0.0412 (95% CI 0.0400 to 0.0424) | Mean squared error of predicted risk. Lower is better. The R value was computed afterwards from the saved R predictions, not reported in the original analysis |
| Mean Predicted Risk | 39.8% | 4.7% | Observed death rate in the holdout is 4.7% |
| Youden Decision Threshold | 0.50 (nominal) / 0.55 | 0.0494 | Cutoff on predicted risk. Tuned out-of-fold in Python |
| Test Sensitivity | 69.1% | 73.2% | Share of deaths flagged high risk |
| Test Specificity | 75.3% | 75.0% | Share of survivors not flagged |
| Test Precision | Not reported | 12.6% | Share of flagged patients who died |
| Test F1-Score | 0.2016 | 0.2146 | Harmonic mean of precision and sensitivity |

The cross-validated AUC (0.7657) is calculated on the training partition, with each model scored on a fold it was not fitted on, and the test AUC (0.8048) is calculated on the separate 25% holdout, so the two come from different data. The 95% intervals come from 2,000 bootstrap resamples of the holdout (3,266 patients, 153 deaths), drawn separately from the deaths and the survivors so that every resample contains deaths, and they show how far a single 25% holdout can move the estimate. The R test AUC lies inside the Python interval.

For reference, a model that gave every patient the same risk of 4.7% would have a Brier score of about 0.045. The R model's Brier score is far above this because of the weighting. An earlier Python run with the same class weighting as R gave a test AUC of 0.8029 and a Brier score of 0.1891, so removing the weighting did not change how well patients are ranked.

At the Python threshold of 0.0494, about 27% of holdout patients are flagged high risk, and about 13% of flagged patients died. The flag is therefore a screening flag.

The calibration curve is stored at `reports/calibration_plot.png`. It sorts the holdout patients into ten equal groups of about 330 by predicted risk and compares the mean predicted risk in each group with the proportion who died. In the highest-risk group the two agree closely (predicted 0.18, observed 0.19). In the groups with predicted risk below about 0.035 the model slightly overestimates risk, with each of these groups containing fewer than 10 deaths. In two middle groups it underestimates (predicted 0.053 and 0.080, observed 0.083 and 0.089), and these groups contain about 27 and 29 deaths. Differences of this size could partly be chance. No recalibration was applied.

---

## Limitations and Governance

1. **Synthetic / Benchmark Data:** The dataset is derived or anonymised without individual patient identifiers. It does not represent validated clinical records from any specific healthcare provider.
2. **No External Validation:** The model has only undergone internal cross-validation and split-sample holdout testing. Generalisability across different clinical settings remains unverified.
3. **Research Demonstration:** This repository is an educational software engineering and machine learning lifecycle demonstration. It must not be deployed in real clinical care or used to influence medical decisions.

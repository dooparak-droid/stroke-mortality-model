"""delay_bands_experiment.py

Would a more flexible treatment of delay (bands, or a smooth curve) improve the model?
Read-only: nothing in the project is changed, and the deployed model is not touched.

Three versions of the full 22-predictor model are compared on the same split, with the same tuning
and the same holdout patients as the real model:
    straight   : delay entered as a number (what the deployed model does)
    bands      : delay entered as five bands (1-6, 7-12, 13-24, 25-36, 37-48 hours)
    curve      : delay entered through a smooth cubic spline

Run from the stroke-mortality-model folder:
    .venv/bin/python analysis/delay_bands_experiment.py
"""

import importlib.util
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from stroke_model.data import TARGET_COLUMN, split_cohort_data
from stroke_model.features import ALL_PREDICTORS

IST = Path(os.environ.get("IST_CSV", "../IST_corrected.csv"))   # not needed here, kept for symmetry
data = pd.read_csv("../ist_stroke_14day.csv")
BAND_EDGES = [0, 6, 12, 24, 36, 48]
BAND_LABELS = ["1-6 h", "7-12 h", "13-24 h", "25-36 h", "37-48 h"]
data["delay_band"] = pd.cut(data.delay, BAND_EDGES, labels=BAND_LABELS).astype(str)

train, test = split_cohort_data(data, test_size=0.25, random_state=123)
y_train, y_test = train[TARGET_COLUMN].to_numpy(), test[TARGET_COLUMN].to_numpy()
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)

OTHER_NUMERIC = ["age", "sbp"]
OTHER_CATEGORICAL = [c for c in ALL_PREDICTORS if c not in ("age", "sbp", "delay")]


def build(delay_treatment):
    transformers = [("num", StandardScaler(), OTHER_NUMERIC),
                    ("cat", OneHotEncoder(handle_unknown="ignore"), OTHER_CATEGORICAL)]
    columns = ALL_PREDICTORS if delay_treatment != "bands" else OTHER_NUMERIC + OTHER_CATEGORICAL + ["delay_band"]
    if delay_treatment == "straight":
        transformers.append(("delay", StandardScaler(), ["delay"]))
    elif delay_treatment == "bands":
        transformers.append(("delay", OneHotEncoder(handle_unknown="ignore"), ["delay_band"]))
    else:   # smooth cubic spline with four knots; constant beyond the data
        transformers.append(("delay", SplineTransformer(n_knots=4, degree=3, extrapolation="constant"), ["delay"]))
    pipeline = Pipeline([("prep", ColumnTransformer(transformers)), ("clf", LogisticRegression(max_iter=2000))])
    search = GridSearchCV(pipeline, {"clf__C": np.logspace(-4, 2, 20)}, scoring="roc_auc", cv=cv, n_jobs=-1)
    search.fit(train[columns], y_train)
    return search.best_score_, search.best_estimator_.predict_proba(test[columns])[:, 1]


results = {name: build(name) for name in ("straight", "bands", "curve")}
base_pred = results["straight"][1]

# Paired bootstrap, drawn separately from the deaths and the survivors
rng = np.random.default_rng(123)
d_idx, s_idx = np.flatnonzero(y_test == 1), np.flatnonzero(y_test == 0)
resamples = [np.concatenate([rng.choice(d_idx, len(d_idx)), rng.choice(s_idx, len(s_idx))]) for _ in range(2000)]
base_boot = np.array([roc_auc_score(y_test[i], base_pred[i]) for i in resamples])

print("1. DISCRIMINATION AND ACCURACY OF THE THREE VERSIONS (same holdout patients)")
print(f"   {'delay entered as':18s}{'CV AUC':>9s}{'holdout AUC':>13s}{'Brier':>9s}{'AUC change vs straight (95% interval)':>42s}")
for name, (cv_auc, pred) in results.items():
    auc = roc_auc_score(y_test, pred)
    if name == "straight":
        change = "reference"
    else:
        boot = np.array([roc_auc_score(y_test[i], pred[i]) for i in resamples]) - base_boot
        low, high = np.percentile(boot, [2.5, 97.5])
        change = f"{auc - roc_auc_score(y_test, base_pred):+.4f}  ({low:+.4f} to {high:+.4f})"
    print(f"   {name:18s}{cv_auc:9.4f}{auc:13.4f}{brier_score_loss(y_test, pred):9.4f}{change:>42s}")

print("\n2. DOES THE STRAIGHT LINE MISFIT BY DELAY BAND? (holdout patients: observed against predicted death rate)")
test = test.assign(band=pd.cut(test.delay, BAND_EDGES, labels=BAND_LABELS), straight=results["straight"][1],
                   bands=results["bands"][1], curve=results["curve"][1])
summary = test.groupby("band", observed=True).agg(
    patients=("death", "size"), deaths=("death", "sum"), observed=("death", "mean"),
    straight=("straight", "mean"), bands=("bands", "mean"), curve=("curve", "mean"))
for col in ("observed", "straight", "bands", "curve"):
    summary[col] = (summary[col] * 100).round(1).astype(str) + "%"
print(summary.to_string())

print("\n3. WHAT EACH VERSION PREDICTS FOR THE REFERENCE PATIENT AT FIVE DELAYS")
BASE = {"age": 72.0, "sbp": 150.0, "gender": "F", "consc": "F", "subtype": "PACS", "treat1": "N", "treat2": "N",
        "wakesym": "N", "atrial": "N", "CT": "Y", "Infarc": "N", "hep24": "N", "asp3": "N", "symptom1": "Y",
        "symptom2": "Y", "symptom3": "N", "symptom4": "N", "symptom5": "N", "symptom6": "N", "symptom7": "N",
        "symptom8": "N"}
print("   (fitted on the training patients only, so the figures differ slightly from the deployed model)")
for name in ("straight", "bands", "curve"):
    columns = ALL_PREDICTORS if name != "bands" else OTHER_NUMERIC + OTHER_CATEGORICAL + ["delay_band"]
    # refit once more with the best settings to be able to score new rows
    transformers = [("num", StandardScaler(), OTHER_NUMERIC), ("cat", OneHotEncoder(handle_unknown="ignore"), OTHER_CATEGORICAL)]
    if name == "straight":
        transformers.append(("delay", StandardScaler(), ["delay"]))
    elif name == "bands":
        transformers.append(("delay", OneHotEncoder(handle_unknown="ignore"), ["delay_band"]))
    else:
        transformers.append(("delay", SplineTransformer(n_knots=4, degree=3, extrapolation="constant"), ["delay"]))
    search = GridSearchCV(Pipeline([("prep", ColumnTransformer(transformers)), ("clf", LogisticRegression(max_iter=2000))]),
                          {"clf__C": np.logspace(-4, 2, 20)}, scoring="roc_auc", cv=cv, n_jobs=-1).fit(train[columns], y_train)
    row = []
    for hours in (3, 10, 20, 30, 45):
        patient = pd.DataFrame([dict(BASE, delay=float(hours))])
        patient["delay_band"] = pd.cut(patient.delay, BAND_EDGES, labels=BAND_LABELS).astype(str)
        row.append(f"{hours:2d} h: {search.predict_proba(patient[columns])[0, 1]:.2%}")
    print(f"   {name:9s} " + "   ".join(row))

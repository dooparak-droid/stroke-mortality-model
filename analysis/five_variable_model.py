"""five_variable_model.py

Two questions, both read-only (nothing in the project is changed):

  Part 1. Which of the 22 predictors actually drive IST's own predicted risk (EXPD14)?
          Measured by how much each one adds to the share of variation explained.
  Part 2. Does our 22-variable model beat a model using only the measures IST's score seems to use?
          Same split, same tuning, same holdout patients as the real model.

Run from the stroke-mortality-model folder:
    .venv/bin/python analysis/five_variable_model.py
"""

import importlib.util
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from stroke_model.data import TARGET_COLUMN, split_cohort_data
from stroke_model.features import ALL_PREDICTORS

IST = Path(os.environ.get("IST_CSV", "../IST_corrected.csv"))
spec = importlib.util.spec_from_file_location("build_dataset", "scripts/build_dataset.py")
build_dataset = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_dataset)

DEFICITS = [f"symptom{i}" for i in range(1, 9)]
NUMERIC = {"age", "sbp", "delay", "n_deficits"}

ist = pd.read_csv(IST, low_memory=False, encoding="latin-1")
data = build_dataset.build_dataset(ist)
data["EXPD14"] = ist.loc[data.index, "EXPD14"]
data["n_deficits"] = (data[DEFICITS] == "Y").sum(axis=1)       # number of the eight deficits present
data = data.reset_index(drop=True)

FIVE = ["consc", "atrial", "gender", "age", "n_deficits"]       # what IST's score appears to use

# ======================================================================================
print("PART 1. WHAT DRIVES IST'S PREDICTED RISK (EXPD14)?")
print("   Share of the variation in logit(EXPD14) explained by different sets of predictors,")
print("   and how much it falls when one of the five is removed (a linear fit of the log-odds).")
p = data.EXPD14.clip(1e-9, 1 - 1e-9)
logit = np.log(p / (1 - p))


def r_squared(columns):
    X = pd.get_dummies(data[columns], drop_first=True).astype(float)
    return LinearRegression().fit(X, logit).score(X, logit)


full = r_squared(ALL_PREDICTORS)
five = r_squared(FIVE)
print(f"   all 22 predictors                       : {full:.4f}")
print(f"   the five measures only                  : {five:.4f}   (conscious state, atrial fibrillation, sex, age, number of deficits)")
print(f"   the other 10 predictors only            : {r_squared([c for c in ALL_PREDICTORS if c not in FIVE and c not in DEFICITS]):.4f}")
print("   Drop one of the five from the five-measure model:")
for column in FIVE:
    print(f"      without {column:11s} -> {r_squared([c for c in FIVE if c != column]):.4f}  (loss {five - r_squared([c for c in FIVE if c != column]):.4f})")

# ======================================================================================
print("\nPART 2. DOES THE 22-VARIABLE MODEL BEAT SMALLER MODELS? (same split, same tuning, same holdout)")
train, test = split_cohort_data(data, test_size=0.25, random_state=123)
y_train, y_test = train[TARGET_COLUMN].to_numpy(), test[TARGET_COLUMN].to_numpy()
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)


def fit_and_score(columns):
    """Ridge logistic regression, penalty tuned by 10-fold CV on AUC, as in train.py. No class weighting."""
    numeric = [c for c in columns if c in NUMERIC]
    categorical = [c for c in columns if c not in NUMERIC]
    transformers = []
    if numeric:
        transformers.append(("num", StandardScaler(), numeric))
    if categorical:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical))
    pipeline = Pipeline([("prep", ColumnTransformer(transformers)), ("clf", LogisticRegression(max_iter=1000))])
    search = GridSearchCV(pipeline, {"clf__C": np.logspace(-4, 2, 20)}, scoring="roc_auc", cv=cv, n_jobs=-1)
    search.fit(train[columns], y_train)
    return search.best_score_, search.best_estimator_.predict_proba(test[columns])[:, 1]


candidates = {
    "A. all 22 predictors (the real model's inputs)": ALL_PREDICTORS,
    "B. five measures (alert, AF, sex, age, deficit count)": FIVE,
    "C. five measures + the eight deficits separately": ["consc", "atrial", "gender", "age"] + DEFICITS,
}
for extra in ["delay", "sbp", "subtype", "Infarc", "CT", "wakesym", "treat1", "treat2", "hep24", "asp3"]:
    candidates[f"   B + {extra}"] = FIVE + [extra]
candidates["D. B + delay + sbp + subtype"] = FIVE + ["delay", "sbp", "subtype"]

results = {name: fit_and_score(columns) for name, columns in candidates.items()}
full_pred = results["A. all 22 predictors (the real model's inputs)"][1]

# Paired bootstrap: the same resampled patients are scored by every model, drawn separately from the
# deaths and the survivors so that every resample contains deaths
rng = np.random.default_rng(123)
deaths_idx, survivors_idx = np.flatnonzero(y_test == 1), np.flatnonzero(y_test == 0)
resamples = [np.concatenate([rng.choice(deaths_idx, len(deaths_idx)), rng.choice(survivors_idx, len(survivors_idx))])
             for _ in range(2000)]
full_auc_boot = np.array([roc_auc_score(y_test[i], full_pred[i]) for i in resamples])

print(f"   holdout: {len(test):,} patients, {int(y_test.sum())} deaths ({y_test.mean():.1%}). IST EXPD14 on the same patients: "
      f"AUC {roc_auc_score(y_test, test.EXPD14):.4f}")
print(f"   {'model':56s}{'columns':>8s}{'CV AUC':>9s}{'holdout':>9s}{'Brier':>8s}{'AUC change vs A (95% interval)':>36s}")
for name, (cv_auc, pred) in results.items():
    auc = roc_auc_score(y_test, pred)
    if name.startswith("A."):
        change = "reference"
    else:
        boot = np.array([roc_auc_score(y_test[i], pred[i]) for i in resamples]) - full_auc_boot
        low, high = np.percentile(boot, [2.5, 97.5])
        change = f"{auc - roc_auc_score(y_test, full_pred):+.4f}  ({low:+.4f} to {high:+.4f})"
    print(f"   {name:56s}{len(candidates[name]):8d}{cv_auc:9.4f}{auc:9.4f}{brier_score_loss(y_test, pred):8.4f}{change:>36s}")

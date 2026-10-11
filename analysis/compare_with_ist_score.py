"""compare_with_ist_score.py

How does our model's estimated risk compare with IST's own 'predicted probability of death at
14 days' (the column EXPD14 in the IST data)?

Everything here is read-only. Nothing in the project is changed.

Run from the stroke-mortality-model folder:
    .venv/bin/python analysis/compare_with_ist_score.py

Caveats, which matter for reading the results:
  * IST's documentation does not say how EXPD14 was calculated. Part 3 tests whether it can be
    reproduced from the 22 predictors alone.
  * EXPD14 was almost certainly built using patients that are in our holdout set, whereas our
    model never saw them. That favours EXPD14 in this comparison.
"""

import importlib.util
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from stroke_model.data import TARGET_COLUMN, load_raw_data, split_cohort_data
from stroke_model.features import ALL_PREDICTORS
from stroke_model.predict import StrokePredictor

IST = Path(os.environ.get("IST_CSV", "../IST_corrected.csv"))
TRAINING_FILE = Path("../ist_stroke_14day.csv")

# Reuse the exact selection rules that built the training file
spec = importlib.util.spec_from_file_location("build_dataset", "scripts/build_dataset.py")
build_dataset = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_dataset)

# --------------------------------------------------------------------------------------
print("1. REBUILD THE SAME PATIENTS, WITH IST'S SCORE ATTACHED")
ist = pd.read_csv(IST, low_memory=False, encoding="latin-1")
built = build_dataset.build_dataset(ist)                       # the same 13,063 patients, same order
built["EXPD14"] = ist.loc[built.index, "EXPD14"]               # row labels line up with the IST file
built["COUNTRY"] = ist.loc[built.index, "COUNTRY"]
saved = load_raw_data(TRAINING_FILE)
same = built[saved.columns].reset_index(drop=True).equals(saved)
print(f"   patients: {len(built):,} | identical to the training file (columns and order): {same}")
print(f"   EXPD14 missing for {int(built.EXPD14.isna().sum())} patients; range {built.EXPD14.min():.3f} to {built.EXPD14.max():.3f}")
assert same, "The rebuilt patients differ from the training file, so the holdout set would not match."

# --------------------------------------------------------------------------------------
print("\n2. COMPARE ON THE SAME HOLDOUT PATIENTS")
_, test = split_cohort_data(built.reset_index(drop=True), test_size=0.25, random_state=123)   # same split as training
predictor = StrokePredictor()
y = test[TARGET_COLUMN].to_numpy()
ours = predictor.pipeline.predict_proba(test[ALL_PREDICTORS])[:, 1]
ist_score = test["EXPD14"].to_numpy()
print(f"   holdout patients: {len(test):,} | deaths: {int(y.sum())} ({y.mean():.1%})")
print(f"   {'':26s}{'AUC':>8s}{'Brier':>9s}{'mean predicted':>17s}")
print(f"   {'our model (v2.0.0)':26s}{roc_auc_score(y, ours):8.4f}{brier_score_loss(y, ours):9.4f}{ours.mean():16.1%}")
print(f"   {'IST EXPD14':26s}{roc_auc_score(y, ist_score):8.4f}{brier_score_loss(y, ist_score):9.4f}{ist_score.mean():16.1%}")
print(f"   {'observed death rate':26s}{'':8s}{'':9s}{y.mean():16.1%}")

# Paired bootstrap of the AUC difference: the same resampled patients are used for both scores,
# drawn separately from deaths and survivors, as in evaluate.bootstrap_metric_intervals
rng = np.random.default_rng(123)
deaths_idx, survivors_idx = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
diffs = []
for _ in range(2000):
    idx = np.concatenate([rng.choice(deaths_idx, len(deaths_idx)), rng.choice(survivors_idx, len(survivors_idx))])
    diffs.append(roc_auc_score(y[idx], ours[idx]) - roc_auc_score(y[idx], ist_score[idx]))
low, high = np.percentile(diffs, [2.5, 97.5])
print(f"   AUC difference (ours minus IST): {roc_auc_score(y, ours) - roc_auc_score(y, ist_score):+.4f}"
      f"  (95% bootstrap interval {low:+.4f} to {high:+.4f})")

# IST's score was not fitted to our split, so it can also be assessed on all patients
print(f"   IST EXPD14 on all {len(built):,} patients: AUC {roc_auc_score(built[TARGET_COLUMN], built.EXPD14):.4f}, "
      f"Brier {brier_score_loss(built[TARGET_COLUMN], built.EXPD14):.4f}")

# --------------------------------------------------------------------------------------
print("\n3. IS EXPD14 A FUNCTION OF THE 22 PREDICTORS ALONE? (a test of the caveat)")
eps = 1e-6
logit = np.log(built.EXPD14.clip(eps, 1 - eps) / (1 - built.EXPD14.clip(eps, 1 - eps)))
X = pd.get_dummies(built[ALL_PREDICTORS], drop_first=True).astype(float)
r2_predictors = LinearRegression().fit(X, logit).score(X, logit)
X_country = pd.concat([X, pd.get_dummies(built["COUNTRY"], prefix="country", drop_first=True).astype(float)], axis=1)
r2_country = LinearRegression().fit(X_country, logit).score(X_country, logit)
print(f"   share of the variation in logit(EXPD14) explained by the 22 predictors      : {r2_predictors:.3f}")
print(f"   ... after also adding country                                                : {r2_country:.3f}")
print("   A value near 1.000 means EXPD14 can be rebuilt from those columns, so it is a baseline score.")
print("   A clearly lower value means it uses something else. That may be other baseline information or")
print("   information recorded after randomisation, and this test cannot tell which.")

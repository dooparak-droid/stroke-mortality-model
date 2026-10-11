"""delay_investigation.py

Why does the estimated risk fall as 'hours from stroke onset to randomisation' (delay) rises?
Everything here is read-only. It loads the saved model and the data and changes nothing.

Run from the stroke-mortality-model folder:
    .venv/bin/python analysis/delay_investigation.py
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from stroke_model.data import TARGET_COLUMN, load_raw_data
from stroke_model.features import ALL_PREDICTORS
from stroke_model.predict import StrokePredictor

DATA = Path("../ist_stroke_14day.csv")                     # the training file
IST = Path(os.environ.get("IST_CSV", "../IST_corrected.csv"))   # original IST file, used only in Part 6
# Download IST_corrected.csv from https://datashare.ed.ac.uk/handle/10283/124 and put it there, or set IST_CSV.

predictor = StrokePredictor()
df = load_raw_data(DATA)

# One reference patient (the README example). Only delay changes in step 1.
BASE = {"age": 72.0, "sbp": 150.0, "delay": 14.0, "gender": "F", "consc": "F", "subtype": "PACS",
        "treat1": "N", "treat2": "N", "wakesym": "N", "atrial": "N", "CT": "Y", "Infarc": "N",
        "hep24": "N", "asp3": "N", "symptom1": "Y", "symptom2": "Y", "symptom3": "N", "symptom4": "N",
        "symptom5": "N", "symptom6": "N", "symptom7": "N", "symptom8": "N"}


def wilson(deaths, n, z=1.96):
    """95% interval for an observed proportion."""
    p = deaths / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return centre - half, centre + half


# --------------------------------------------------------------------------------------
print("1. MODEL OUTPUT as only delay changes (all other values fixed at the reference patient)")
for hours in (1, 6, 12, 14, 24, 36, 48, 72, 100):
    risk = predictor.predict_record(dict(BASE, delay=float(hours)))["mortality_probability"]
    note = "   <- outside the training data (1 to 48 h)" if hours > 48 else ""
    print(f"   delay {hours:3d} h -> estimated risk {risk:.4f}{note}")

# --------------------------------------------------------------------------------------
print("\n2. THE MODEL'S COEFFICIENT FOR DELAY")
pre = predictor.pipeline.named_steps["preprocessor"]
clf = predictor.pipeline.named_steps["classifier"]
names = list(pre.get_feature_names_out())
coefs = dict(zip(names, clf.coef_[0]))
# The model standardised delay before fitting. Find the scaler that handled it, so the coefficient
# can be converted from 'per standard deviation' back to 'per hour'.
for _, transformer, columns in pre.transformers_:
    if "delay" in list(columns):
        scale = transformer.scale_[list(columns).index("delay")]
        break
per_hour = coefs["num__delay"] / scale
print(f"   standardised coefficient (log-odds per 1 standard deviation of delay): {coefs['num__delay']:+.3f}")
print(f"   standard deviation of delay: {scale:.1f} hours")
print(f"   per hour: {per_hour:+.4f} log-odds  ->  odds multiplied by {np.exp(per_hour * 12):.3f} for every extra 12 hours")
print("   For comparison, the standardised coefficients of the largest predictors:")
for name, value in sorted(coefs.items(), key=lambda kv: -abs(kv[1]))[:5]:
    print(f"      {name:20s} {value:+.3f}")

# --------------------------------------------------------------------------------------
print("\n3. OBSERVED DEATH RATE BY DELAY BAND (the real data; no model; all 13,063 patients)")
df["band"] = pd.cut(df.delay, [0, 6, 12, 24, 36, 48], labels=["1-6 h", "7-12 h", "13-24 h", "25-36 h", "37-48 h"])
for band, group in df.groupby("band", observed=True):
    n, deaths = len(group), int(group[TARGET_COLUMN].sum())
    low, high = wilson(deaths, n)
    print(f"   {band:8s} patients {n:5d}  deaths {deaths:4d}  rate {deaths / n:6.1%}  (95% interval {low:.1%} to {high:.1%})")

# --------------------------------------------------------------------------------------
print("\n4. DO OTHER PATIENT CHARACTERISTICS CHANGE WITH DELAY? (mean by delay band)")
df["drowsy"] = (df.consc == "D").astype(int)
df["tacs"] = (df.subtype == "TACS").astype(int)
print(df.groupby("band", observed=True)[["age", "sbp", "drowsy", "tacs"]].mean().round(3).to_string())

# --------------------------------------------------------------------------------------
print("\n5. DOES THE NEGATIVE DELAY EFFECT SURVIVE ADJUSTMENT? (three ordinary logistic regressions)")
print("   Odds multiplier per extra 12 hours of delay. Below 1 means lower odds of death with longer delay.")


def delay_odds_multiplier(columns):
    """Fit a logistic regression (essentially unpenalised) and return the delay effect per 12 hours."""
    X = pd.get_dummies(df[columns], drop_first=True).astype(float)
    X["delay"] = X["delay"] / 12                     # so the coefficient is per 12 hours
    if "age" in X:
        X["age"] = X["age"] / 10
    if "sbp" in X:
        X["sbp"] = X["sbp"] / 10
    model = LogisticRegression(C=1e6, max_iter=5000).fit(X, df[TARGET_COLUMN])
    return float(np.exp(model.coef_[0][list(X.columns).index("delay")]))


print(f"   (a) delay alone                                     : {delay_odds_multiplier(['delay']):.3f}")
print(f"   (b) delay + drowsy + total anterior stroke          : {delay_odds_multiplier(['delay', 'drowsy', 'tacs']):.3f}")
print(f"   (c) delay + all 21 other predictors (the full model): {delay_odds_multiplier(ALL_PREDICTORS):.3f}")

# --------------------------------------------------------------------------------------
print("\n6. WHEN DOES THE 14-DAY CLOCK START? (needs the IST download)")
if IST.is_file():
    ist = pd.read_csv(IST, low_memory=False, encoding="latin-1")
    died = ist[(ist.ID14 == 1) & ist.RDELAY.notna() & ist.TD.notna()].copy()
    died["band"] = pd.cut(died.RDELAY, [0, 6, 12, 24, 36, 48], labels=["1-6 h", "7-12 h", "13-24 h", "25-36 h", "37-48 h"])
    print("   Days to death (IST variable TD) among patients who died within 14 days, by delay band:")
    print(died.groupby("band", observed=True).TD.agg(deaths="size", shortest="min", median="median", longest="max").to_string())
    print("   If days were counted from stroke onset, patients randomised 25 to 48 hours after onset")
    print("   could not die at day 0. If counted from randomisation, they can.")
else:
    print(f"   IST file not found at {IST}. Set IST_CSV to its path to run this part.")

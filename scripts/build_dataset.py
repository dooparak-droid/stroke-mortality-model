"""Build the training file from the International Stroke Trial (IST) database.

The IST data are not stored in this repository. Download `IST_corrected.csv` from the
University of Edinburgh DataShare record (https://doi.org/10.7488/ds/104), then run:

    python scripts/build_dataset.py --ist-csv path/to/IST_corrected.csv --out ../ist_stroke_14day.csv

Selection of patients
    1. Alert (F) or drowsy (D) at randomisation. Patients recorded as unconscious (U) are excluded.
    2. A recorded value for each of the 22 predictors. This also excludes patients with any
       of the eight neurological deficit variables coded C ("cannot assess").
    3. A known death status on the discharge form (DDEAD is Y or N). This keeps the same
       13,063 patients as the coursework version of the file.

Outcome
    death = 1 if the IST 14-day death indicator ID14 is 1, otherwise 0. ID14 counts every
    death within 14 days of randomisation, whether or not it was recorded on the discharge
    form. The script stops if any selected patient has an unknown 14-day status (SET14D = 0).
"""

import argparse
from pathlib import Path

import pandas as pd

# IST variable name -> column name used by the model
PREDICTOR_MAP = {
    "RDELAY": "delay",
    "RCONSC": "consc",
    "SEX": "gender",
    "AGE": "age",
    "RSLEEP": "wakesym",
    "RATRIAL": "atrial",
    "RCT": "CT",
    "RVISINF": "Infarc",
    "RHEP24": "hep24",
    "RASP3": "asp3",
    "RSBP": "sbp",
    "RDEF1": "symptom1",
    "RDEF2": "symptom2",
    "RDEF3": "symptom3",
    "RDEF4": "symptom4",
    "RDEF5": "symptom5",
    "RDEF6": "symptom6",
    "RDEF7": "symptom7",
    "RDEF8": "symptom8",
    "STYPE": "subtype",
    "RXHEP": "treat1",
    "RXASP": "treat2",
}
DEFICIT_COLUMNS = [f"RDEF{i}" for i in range(1, 9)]
OUTPUT_COLUMNS = list(PREDICTOR_MAP.values()) + ["death"]


def build_dataset(ist: pd.DataFrame) -> pd.DataFrame:
    """Apply the selection rules and outcome definition to the IST table."""
    selected = ist[ist["RCONSC"].isin(["F", "D"])]
    selected = selected.dropna(subset=list(PREDICTOR_MAP))
    selected = selected[selected[DEFICIT_COLUMNS].isin(["Y", "N"]).all(axis=1)]
    selected = selected[selected["DDEAD"].isin(["Y", "N"])].copy()

    if (selected["SET14D"] != 1).any():
        raise ValueError("Some selected patients have an unknown 14-day status (SET14D = 0).")

    selected["death"] = selected["ID14"].astype(int)
    return selected.rename(columns=PREDICTOR_MAP)[OUTPUT_COLUMNS]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ist-csv", required=True, help="Path to IST_corrected.csv")
    parser.add_argument("--out", required=True, help="Where to write the training file")
    args = parser.parse_args()

    # IST_corrected.csv is not UTF-8 encoded
    ist = pd.read_csv(args.ist_csv, low_memory=False, encoding="latin-1")
    dataset = build_dataset(ist)

    out_path = Path(args.out)
    dataset.to_csv(out_path, index=False)
    print(f"Wrote {len(dataset):,} patients to {out_path}")
    print(f"14-day deaths: {int(dataset['death'].sum()):,} ({dataset['death'].mean():.1%})")


if __name__ == "__main__":
    main()

# Analysis scripts

Read-only scripts that investigate the model and the data. None of them changes the model, the data or any file in the project. They support the statements in the main README about the delay field and about benchmarks.

They need the full IST download, because they use columns that the training file does not contain. Download `IST_corrected.csv` from the [DataShare record](https://datashare.ed.ac.uk/handle/10283/124) and place it beside `ist_stroke_14day.csv`, or set `IST_CSV` to its location. The data are not committed to this repository.

Run each one from the project folder:

| Script | Question it answers | Takes about |
|---|---|---|
| `analysis/delay_investigation.py` | Why does the estimated risk fall as the delay from onset to randomisation rises? Does the effect survive adjustment? When does the 14-day clock start? | 10 seconds |
| `analysis/compare_with_ist_score.py` | How does the model compare with IST's own predicted probability of death at 14 days (`EXPD14`) on the same holdout patients? | 15 seconds |
| `analysis/five_variable_model.py` | Does the 22-variable model beat a model using only conscious state, atrial fibrillation, sex, age and the number of deficits? | 1 minute |
| `analysis/delay_bands_experiment.py` | Would entering delay as bands, or as a smooth curve, improve the model? | 1 minute |

```bash
.venv/bin/python analysis/compare_with_ist_score.py
```

All of them use the same split (seed 123) and the same tuning as the training script, so they score the same holdout patients as the deployed model.

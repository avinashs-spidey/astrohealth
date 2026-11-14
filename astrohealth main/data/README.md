# ASTROHEALTH Training & Testing Datasets

All data is synthetic (simulated astronaut physiological and radiation telemetry) for training and evaluating the ASTROHEALTH risk classifier and anomaly detector. Not a medical device.

## Files
- `train_telemetry.csv` (6,720 rows): Labeled training dataset for the `RandomForestClassifier` risk model.
- `test_telemetry.csv` (1,680 rows): Held-out labeled test dataset used by `GET /api/model/eval` to compute accuracy, precision/recall/F1, and confusion matrix.
- `baseline_nominal.csv` (3,000 rows): Nominal baseline telemetry used to fit the `IsolationForest` anomaly detector.
- `generate_datasets.py`: Reproducible generator script (`python data/generate_datasets.py`).

## Columns
- Features (`14`): `hr`, `sys`, `dia`, `spo2`, `temp`, `sleep`, `stress`, `fatigue`, `mood`, `bone_loss`, `muscle_loss`, `wbc`, `crp`, `rad`
- Labels (`train_telemetry.csv`, `test_telemetry.csv`):
  - `risk_class`: `0` (`low`), `1` (`moderate`), `2` (`high`)
  - `risk_label`: `low` | `moderate` | `high`
  - `is_anomaly`: `0` | `1`

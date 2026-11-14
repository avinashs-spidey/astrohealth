# ASTROHEALTH (simulated, software-only)

Holographic orbital bio-telemetry, EVA readiness, SOS distress dispatch, and ML health monitoring dashboard. All data is simulated. Not a medical device.

## Setup & Run
```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python.exe -m uvicorn main:app --reload   # open http://127.0.0.1:8000
```

## Features
- **Mission Simulation**: Seeds 4 astronauts across a 180-day mission (nominal, cardiovascular event, immune shift, psychological decline) plus a solar particle event on days 120–122 and Scenario 4 (multi-system EVA strain).
- **Holographic Bio-HUD & Multi-System Telemetry**: Live bio-suit hologram SVG, radial gauges, 6-axis organ strain radar, 15-biomarker limits table, and 6 time-series charts.
- **EVA Readiness & Countermeasures**: Go/Caution/No-Go advisor across 5 domains and interactive 14-day intervention simulator (`sleep_reset`, `exercise_boost`, `storm_shelter`, `medical_stabilize`, `inject_solar_event`).
- **Space Location Tagging & SOS Beacon**: Orbital/surface crew position map, zone/module/coordinate tagging with light-time comms delay, and `MAYDAY`/`URGENT` SOS distress beacon dispatch & resolution log.
- **ML Training & Testing Data (`data/`)**: Trains `RandomForestClassifier` on `data/train_telemetry.csv` (6,720 rows) and `IsolationForest` on `data/baseline_nominal.csv` (3,000 rows), with held-out evaluation on `data/test_telemetry.csv` (1,680 rows) via `/api/model/eval`.
- **Personalized Baseline, Forecasting & Export**: Per-astronaut z-score drift monitor, threshold ETA forecasting, interactive Flight Surgeon Q&A (`/api/astronauts/{id}/ask`), clinical notes, plain-text report, and CSV export.

PostgreSQL: `pip install psycopg2-binary` and set `DATABASE_URL`.
Offline: only the Chart.js CDN script needs internet; download it into `static/` for fully offline use.
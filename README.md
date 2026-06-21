# F1 Qualifying Predictor

A Flask web app that predicts the qualifying grid for the next Formula 1 Grand Prix. It fetches real session data via [FastF1](https://github.com/theOehrly/Fast-F1), engineers 16 race-relevant features, trains an XGBoost regressor, and streams live progress to the browser while the model runs.

## Features

- **Live progress streaming** — SSE-based loading page shows each data-fetch and training step in real time
- **Circuit-aware predictions** — each driver's pace is normalised to the next circuit's reference lap time, so raw seconds aren't averaged across tracks of different lengths
- **Session quality weighting** — year recency discount (0.5× per year back), wet sessions (0.4×), sprint weekends (0.5×)
- **Rookie fallback** — anchors to teammate's last 5 races with a 0.5% deficit; Q3 rate set to 0.15
- **Confidence intervals** — std of predictions across 20 XGBoost tree-slice checkpoints
- **Shareable results** — predictions cached server-side, accessible via `/results?id=<hash>`
- **Prediction history** — append-only JSON log at `data/accuracy_log.json`, viewable at `/history`
- **Model caching** — joblib-serialised bundle keyed by `MODEL_VERSION` + data fingerprint; stale caches auto-deleted

## Stack

| Layer | Library |
|---|---|
| Data | FastF1, pandas |
| Model | XGBoost, scikit-learn (StandardScaler, SimpleImputer) |
| Backend | Flask |
| Templates | Jinja2 + vanilla JS |

## Model Architecture

**Target:** raw Q3 lap time in seconds (not normalised)

**Training data:** 3 seasons (current − 2, current − 1, current), all completed qualifying sessions fetched via FastF1.

**Features (16 total):**

| Feature | Description |
|---|---|
| `Q1_sec`, `Q2_sec` | Raw lap times in seconds |
| `Q1_Q2_diff`, `Q1_Q2_improvement_pct` | Q1→Q2 delta |
| `Q1_normalized`, `Q2_normalized` | Per-circuit normalised pace |
| `Driver_Q1_ewm`, `Driver_Q2_ewm` | EWM rolling average (span=3) per driver, cross-year |
| `Team_Q1_mean`, `Team_Q2_mean` | Cross-year team pace per round |
| `Team_Q1_current_year`, `Team_Q2_current_year` | Current-season team pace (normalised), isolated from prior years |
| `Team_Q3_current_year` | Current-season team Q3 mean in raw seconds |
| `TrackType_enc` | 0 = street, 1 = technical, 2 = high-speed |
| `Driver_Q3_rate` | Historical Q3 participation rate per driver |
| `Circuit_Q3_mean` | Per-circuit historical Q3 mean — anchors the model to the correct absolute lap time scale |

**XGBoost config:** 400 estimators, learning rate 0.05, max depth 6, subsample 0.8, colsample_bytree 0.8.

`Circuit_Q3_mean` is included as a feature (not used to normalise the target) so the model learns the absolute time scale per track while keeping R² interpretable (~0.998 on held-out data).

## Project Structure

```
qualiprediction.py      — all backend logic (data, features, model, routes)
templates/
  index.html            — home page
  loading.html          — live progress (SSE)
  results.html          — predictions table with metrics and share link
  history.html          — collapsible log of past predictions
data/
  rosters.yaml          — driver/team rosters by year (2025, 2026)
  accuracy_log.json     — append-only prediction log
model_cache/            — joblib model bundles + shareable result JSON
```

## Setup

```bash
pip install -r requirements.txt
python qualiprediction.py
```

Open `http://localhost:5000` and click **Predict** to run.

FastF1 downloads session data on first use and caches it under `cache/`. Subsequent runs for the same season are much faster.

## Model Versioning

`MODEL_VERSION` is set at the top of `qualiprediction.py`. Bump it and delete `model_cache/*.joblib` whenever you change features, model hyperparameters, or prediction logic — the cache fingerprint will mismatch and a fresh model will be trained automatically.

## Design System

Black background (`#0a0a0a`), white text, red (`#e8001d`) accent only. Inter for body text, Jeko Sans for headings. Material Symbols Outlined for icons. No Bootstrap, no gradients.

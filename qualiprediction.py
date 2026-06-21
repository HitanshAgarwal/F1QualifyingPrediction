import fastf1
import pandas as pd
import numpy as np
import yaml
import json
import time
import joblib
import hashlib
import os
import warnings
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from flask import Flask, render_template, request, jsonify, Response, redirect, url_for
warnings.filterwarnings('ignore')

app = Flask(__name__)
fastf1.Cache.enable_cache('cache')

CACHE_DIR = Path('model_cache')
CACHE_DIR.mkdir(exist_ok=True)

ACCURACY_LOG = Path('data/accuracy_log.json')
ACCURACY_LOG.parent.mkdir(exist_ok=True)

ROSTER_FILE = Path('data/rosters.yaml')

# ---------------------------------------------------------------------------
# Roster loading (improvement #4 — config file, #2 — dynamic FastF1 fallback)
# ---------------------------------------------------------------------------

def load_rosters_from_yaml():
    with open(ROSTER_FILE) as f:
        data = yaml.safe_load(f)
    return {int(k): v for k, v in data['rosters'].items()}

def get_driver_teams_dynamic(year):
    """Try to fetch current season roster from FastF1, fall back to YAML."""
    try:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        completed = [r for _, r in schedule.iterrows()
                     if pd.Timestamp(r['EventDate']) < pd.Timestamp.now()]
        if not completed:
            raise ValueError("No completed rounds yet")
        last_round = completed[-1]['RoundNumber']
        session = fastf1.get_session(year, int(last_round), 'Q')
        session.load(laps=False, telemetry=False, weather=False, messages=False)
        results = session.results
        roster = {row['FullName']: row['TeamName'] for _, row in results.iterrows()}
        if len(roster) >= 18:
            return roster
    except Exception as e:
        print(f"Dynamic roster fetch failed ({e}), falling back to YAML")

    rosters = load_rosters_from_yaml()
    return rosters.get(year, rosters[max(rosters.keys())])

def get_driver_teams(year):
    return get_driver_teams_dynamic(year)

# Drivers whose FastF1 full name varies across seasons or rounds.
# Maps any known variant -> canonical name used in rosters.yaml.
DRIVER_NAME_ALIASES = {
    'Andrea Kimi Antonelli': 'Kimi Antonelli',
    'Andrea Kimi Antonelli ': 'Kimi Antonelli',
}

def normalize_driver_name(name):
    if not isinstance(name, str):
        return name
    return DRIVER_NAME_ALIASES.get(name.strip(), name.strip())

# ---------------------------------------------------------------------------
# Data fetching (improvement #3 — per-round error recovery)
# ---------------------------------------------------------------------------

def fetch_f1_data(year, round_number):
    """Fetch qualifying data for one round. Returns None silently on failure."""
    try:
        quali = fastf1.get_session(year, round_number, 'Q')
        quali.load(laps=False, telemetry=False, weather=True, messages=False)

        results = quali.results[['DriverNumber', 'FullName', 'TeamName', 'Q1', 'Q2', 'Q3', 'Position']].copy()
        results = results.rename(columns={'FullName': 'Driver'})
        results['Driver'] = results['Driver'].apply(normalize_driver_name)

        for col in ['Q1', 'Q2', 'Q3']:
            results[col + '_sec'] = results[col].apply(
                lambda x: x.total_seconds() if pd.notnull(x) else None
            )

        results['Year'] = year
        results['Round'] = round_number
        results['TrackName'] = quali.event['EventName']
        results['CircuitKey'] = str(quali.event.get('OfficialEventName', quali.event['EventName']))

        # Improvement #7 — flag sessions that are likely anomalous
        # Sprint weekends: qualifying session is Q only (no Q2/Q3) OR session type differs
        is_sprint_weekend = getattr(quali.event, 'EventFormat', '') in ('sprint', 'sprint_shootout')
        results['IsSprint'] = is_sprint_weekend

        # Wet session heuristic: >30% of drivers with Q1 times more than 5% above track median
        if results['Q1_sec'].notna().sum() >= 5:
            median_q1 = results['Q1_sec'].median()
            wet_fraction = (results['Q1_sec'] > median_q1 * 1.05).mean()
            results['IsWet'] = wet_fraction > 0.3
        else:
            results['IsWet'] = False

        return results

    except Exception as e:
        print(f"  Skipped {year} R{round_number}: {e}")
        return None

# ---------------------------------------------------------------------------
# Feature engineering (improvements #5, #6, #8, #9)
# ---------------------------------------------------------------------------

TRACK_TYPES = {
    # street circuits
    'Monaco Grand Prix': 'street',
    'Azerbaijan Grand Prix': 'street',
    'Singapore Grand Prix': 'street',
    'Saudi Arabian Grand Prix': 'street',
    'Las Vegas Grand Prix': 'street',
    'Miami Grand Prix': 'street',
    'Abu Dhabi Grand Prix': 'street',
    # high-speed
    'Italian Grand Prix': 'highspeed',
    'Belgian Grand Prix': 'highspeed',
    'British Grand Prix': 'highspeed',
    'Austrian Grand Prix': 'highspeed',
    # default: technical
}

def get_track_type(track_name):
    for key, t in TRACK_TYPES.items():
        if key in track_name:
            return t
    return 'technical'

def exponential_weighted_rolling(series, span=3):
    """EWM mean per driver ordered by appearance."""
    return series.ewm(span=span, adjust=False).mean()

def engineer_features(df):
    df = df.copy()

    # Basic time features
    df['Q1_Q2_diff'] = df['Q2_sec'] - df['Q1_sec']
    df['Q1_Q2_improvement_pct'] = ((df['Q1_sec'] - df['Q2_sec']) / df['Q1_sec'] * 100).where(df['Q1_sec'] > 0)

    # Improvement #5 — circuit type as ordinal
    df['TrackType'] = df['TrackName'].apply(get_track_type)
    track_type_map = {'street': 0, 'technical': 1, 'highspeed': 2}
    df['TrackType_enc'] = df['TrackType'].map(track_type_map).fillna(1)

    # Improvement #6 — circuit-specific normalization (normalize within each unique track)
    for track in df['TrackName'].unique():
        mask = df['TrackName'] == track
        q1_mean = df.loc[mask, 'Q1_sec'].mean()
        q2_mean = df.loc[mask, 'Q2_sec'].mean()
        if q1_mean and q1_mean > 0:
            df.loc[mask, 'Q1_normalized'] = df.loc[mask, 'Q1_sec'] / q1_mean
        if q2_mean and q2_mean > 0:
            df.loc[mask, 'Q2_normalized'] = df.loc[mask, 'Q2_sec'] / q2_mean

    # Improvement #8 — exponential decay rolling averages (EWM, span=3)
    df = df.sort_values(['Driver', 'Year', 'Round'])
    df['Driver_Q1_ewm'] = df.groupby('Driver')['Q1_sec'].transform(exponential_weighted_rolling)
    df['Driver_Q2_ewm'] = df.groupby('Driver')['Q2_sec'].transform(exponential_weighted_rolling)

    # Team performance per round (cross-year)
    df['Team_Q1_mean'] = df.groupby(['TeamName', 'Round'])['Q1_sec'].transform('mean')
    df['Team_Q2_mean'] = df.groupby(['TeamName', 'Round'])['Q2_sec'].transform('mean')

    # Current-year team pace — isolated from historical seasons so car development
    # changes aren't diluted by prior year performance
    current_year = df['Year'].max()
    current_year_mask = df['Year'] == current_year
    for seg, col in [('Q1', 'Q1_normalized'), ('Q2', 'Q2_normalized'), ('Q3', 'Q3_sec')]:
        feat = f'Team_{seg}_current_year'
        team_current = (
            df[current_year_mask]
            .groupby('TeamName')[col]
            .mean()
            .rename(feat)
        )
        df = df.join(team_current, on='TeamName')
        prior_mask = ~current_year_mask & df[feat].isna()
        if prior_mask.any():
            prior_pace = df[prior_mask].groupby(['TeamName', 'Year'])[col].transform('mean')
            df.loc[prior_mask, feat] = prior_pace

    # Improvement #9 — Q3 participation rate per driver (overall historical)
    df['MadeQ3'] = df['Q3_sec'].notna().astype(float)
    driver_q3_rate = df.groupby('Driver')['MadeQ3'].transform('mean')
    df['Driver_Q3_rate'] = driver_q3_rate

    # Circuit Q3 mean — lets the model learn absolute lap time scale per track
    # computed per TrackName across all years so it's stable
    circuit_q3_mean = df.groupby('TrackName')['Q3_sec'].transform('mean')
    df['Circuit_Q3_mean'] = circuit_q3_mean

    # --- Additional driver features ---

    # Recent form trend: slope of Q1_normalized over last 5 rounds per driver.
    # Positive = improving (going faster relative to field), negative = fading.
    def _form_trend(series):
        out = pd.Series(np.nan, index=series.index)
        vals = series.values
        for i in range(len(vals)):
            window = vals[max(0, i - 4): i + 1]
            valid = window[~np.isnan(window)]
            if len(valid) >= 3:
                x = np.arange(len(valid))
                out.iloc[i] = float(np.polyfit(x, valid, 1)[0])
        return out

    df['Driver_form_trend'] = df.groupby('Driver')['Q1_normalized'].transform(_form_trend)

    # Pace consistency: rolling std of Q1_normalized (span=5) per driver.
    # Low std = predictable; high std = erratic. Imputed with driver mean on NaN.
    def _rolling_std(series, window=5):
        return series.rolling(window, min_periods=2).std()

    df['Driver_pace_consistency'] = df.groupby('Driver')['Q1_normalized'].transform(_rolling_std)

    # Teammate delta: driver Q1_normalized minus teammate Q1_normalized in same session.
    # Negative = faster than teammate; positive = slower.
    teammate_mean = df.groupby(['TeamName', 'Round', 'Year'])['Q1_normalized'].transform('mean')
    df['Teammate_Q1_delta'] = df['Q1_normalized'] - teammate_mean

    # Q2 form trend: slope of Q2_normalized over last 5 sessions per driver.
    # Q2 trend captures a driver's improvement under pressure (Q2 is where
    # mid-field drivers are most stressed), distinct from the Q1 slope.
    df['Driver_Q2_trend'] = df.groupby('Driver')['Q2_normalized'].transform(_form_trend)

    # Session weight: year recency discount × session quality factor
    # Current year = 1.0, one year back = 0.5, two years back = 0.25
    max_year = df['Year'].max()
    year_discount = 0.5 ** (max_year - df['Year'])
    df['SessionWeight'] = year_discount
    df.loc[df['IsWet'], 'SessionWeight'] *= 0.4
    df.loc[df['IsSprint'], 'SessionWeight'] *= 0.5

    return df

# ---------------------------------------------------------------------------
# Model training (improvement #10 — XGBoost; #1 — joblib caching)
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    'Q1_sec', 'Q2_sec', 'Q1_Q2_diff', 'Q1_Q2_improvement_pct',
    'Q1_normalized', 'Q2_normalized',
    'Driver_Q1_ewm', 'Driver_Q2_ewm',
    'Team_Q1_mean', 'Team_Q2_mean',
    'Team_Q1_current_year', 'Team_Q2_current_year', 'Team_Q3_current_year',
    'TrackType_enc', 'Driver_Q3_rate',
    'Circuit_Q3_mean',
    # Driver-level features
    'Driver_form_trend',       # recent Q1 pace slope (positive = improving)
    'Driver_Q2_trend',         # recent Q2 pace slope (Q2 pressure performance)
    'Driver_pace_consistency', # rolling std of normalised pace (lower = steadier)
    'Teammate_Q1_delta',       # pace vs teammate in same session (negative = faster)
]

# Bump this whenever model logic, features, or prediction code changes.
MODEL_VERSION = "12"

def _data_fingerprint(df):
    """Hash based on data shape/range + model version + today's date.
    Cache refreshes daily so form trends stay current even if no new round has been added."""
    today = pd.Timestamp.now().strftime('%Y-%m-%d')
    key = (f"v{MODEL_VERSION}_{df.shape[0]}_{df['Year'].min()}_{df['Year'].max()}"
           f"_{df['Round'].max()}_{df['Driver'].nunique()}_{today}")
    return hashlib.md5(key.encode()).hexdigest()[:12]

def train_model(df, yield_status=None):
    fingerprint = _data_fingerprint(df)
    cache_path = CACHE_DIR / f"model_{fingerprint}.joblib"

    if cache_path.exists():
        if yield_status:
            yield_status('training', 'Loading cached model (data unchanged)...')
        bundle = joblib.load(cache_path)
        return bundle

    if yield_status:
        yield_status('training', 'Training XGBoost model...')

    df_clean = df.dropna(subset=['Q3_sec']).copy()
    df_clean = df_clean[df_clean['SessionWeight'] > 0.3]

    X = df_clean[FEATURE_COLS].copy()
    y = df_clean['Q3_sec'].copy()
    weights = df_clean['SessionWeight'].copy()

    imputer = SimpleImputer(strategy='median', keep_empty_features=True)
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=FEATURE_COLS, index=X.index)

    X_train, X_test, y_train, y_test, w_train, _ = train_test_split(
        X_imp, y, weights, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_train_s, y_train, sample_weight=w_train)

    y_pred = model.predict(X_test_s)
    mae  = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2   = float(r2_score(y_test, y_pred))

    print(f"\n=== Model Performance ===")
    print(f"Test MAE:  {mae:.3f}s  |  RMSE: {rmse:.3f}s  |  R²: {r2:.3f}")

    bundle = dict(
        model=model, scaler=scaler, imputer=imputer,
        mae=mae, rmse=rmse, r2=r2,
        fingerprint=fingerprint,
    )
    joblib.dump(bundle, cache_path)
    # Remove stale caches (keep only latest)
    for old in CACHE_DIR.glob('model_*.joblib'):
        if old.name != cache_path.name:
            old.unlink(missing_ok=True)

    return bundle

# ---------------------------------------------------------------------------
# Season / race helpers
# ---------------------------------------------------------------------------

def get_current_season():
    return pd.Timestamp.now().year

def get_training_years():
    current = get_current_season()
    return [current - 2, current - 1, current]

def get_next_race_info():
    current_year = get_current_season()
    today = pd.Timestamp.now()
    for year in [current_year, current_year + 1]:
        try:
            schedule = fastf1.get_event_schedule(year, include_testing=False)
            future = schedule[pd.to_datetime(schedule['EventDate']) > today]
            if len(future) > 0:
                nxt = future.iloc[0]
                return {
                    'round': int(nxt['RoundNumber']),
                    'name': str(nxt['EventName']),
                    'location': str(nxt['Location']),
                    'date': str(nxt['EventDate']),
                    'year': year,
                }
        except Exception as e:
            print(f"Schedule fetch error {year}: {e}")
    return {'round': None, 'name': 'Next Grand Prix', 'location': 'TBD', 'date': None, 'year': current_year}

# ---------------------------------------------------------------------------
# Prediction (improvement #12 — confidence intervals via XGBoost individual trees)
# ---------------------------------------------------------------------------

def predict_next_race(bundle, historical_data, next_race, season_year=None):
    model = bundle['model']
    scaler = bundle['scaler']
    imputer = bundle['imputer']

    if season_year is None:
        season_year = get_current_season()
    driver_teams = get_driver_teams(season_year)

    next_track = next_race.get('name', '')
    next_track_type = get_track_type(next_track)
    track_type_map = {'street': 0, 'technical': 1, 'highspeed': 2}
    next_track_type_enc = track_type_map.get(next_track_type, 1)

    # Circuit-specific reference: use same-track history if available
    track_data = historical_data[historical_data['TrackName'].str.contains(
        next_track.split(' Grand')[0], case=False, na=False
    )]
    if len(track_data) >= 5:
        ref_q1_mean = track_data['Q1_sec'].mean()
        ref_q2_mean = track_data['Q2_sec'].mean()
        circuit_q3_mean = track_data['Q3_sec'].dropna().mean()
    else:
        ref_q1_mean = historical_data['Q1_sec'].mean()
        ref_q2_mean = historical_data['Q2_sec'].mean()
        circuit_q3_mean = historical_data['Q3_sec'].dropna().mean()

    # Build a teammate lookup: for each driver, find the other driver at the same team
    team_to_drivers = {}
    for d, t in driver_teams.items():
        team_to_drivers.setdefault(t, []).append(d)

    predictions = []
    for driver, team in driver_teams.items():
        # Split history: with current team vs any team.
        # A driver who switched teams (e.g. Ocon Alpine→Haas) has rich personal
        # history but it reflects a different car — using it raw would inflate pace.
        driver_hist_team = historical_data[
            (historical_data['Driver'] == driver) &
            (historical_data['TeamName'] == team)
        ].tail(5)
        driver_hist_any = historical_data[historical_data['Driver'] == driver].tail(5)
        team_hist = historical_data[historical_data['TeamName'] == team].tail(15)

        if len(driver_hist_team) > 0:
            # Happy path: driver has raced for this team — use that history directly.
            avg_q1_norm = driver_hist_team['Q1_normalized'].mean()
            avg_q2_norm = driver_hist_team['Q2_normalized'].mean()
            avg_q1 = avg_q1_norm * ref_q1_mean
            avg_q2 = avg_q2_norm * ref_q2_mean
            avg_q1_ewm = driver_hist_team['Driver_Q1_ewm'].iloc[-1] / driver_hist_team['Q1_sec'].mean() * avg_q1
            avg_q2_ewm = driver_hist_team['Driver_Q2_ewm'].iloc[-1] / driver_hist_team['Q2_sec'].mean() * avg_q2
            driver_q3_rate = driver_hist_team['MadeQ3'].mean() if 'MadeQ3' in driver_hist_team.columns else 0.5

        elif len(driver_hist_any) > 0 and len(team_hist) > 0:
            # Team-switch case: driver has history but at a different team.
            # Blend the driver's relative pace rank (from old team) with the new
            # team's absolute pace level so the car context dominates.
            # driver_rel = how fast the driver is vs the field (normalised)
            # team_abs   = what absolute pace that field-relative speed maps to at the new team
            driver_rel_q1 = driver_hist_any['Q1_normalized'].mean()   # e.g. 1.002 = 0.2% above avg
            driver_rel_q2 = driver_hist_any['Q2_normalized'].mean()
            team_abs_q1   = team_hist['Q1_normalized'].mean() * ref_q1_mean
            team_abs_q2   = team_hist['Q2_normalized'].mean() * ref_q2_mean
            # Scale: team's absolute pace × driver's relative rank within their old team context
            avg_q1 = team_abs_q1 * driver_rel_q1
            avg_q2 = team_abs_q2 * driver_rel_q2
            avg_q1_ewm = avg_q1
            avg_q2_ewm = avg_q2
            # Q3 rate anchored to new team's historical rate — old team's rate is irrelevant
            team_q3_rate = (team_hist['MadeQ3'].mean()
                            if 'MadeQ3' in team_hist.columns else 0.3)
            driver_q3_rate = team_q3_rate

        else:
            # Rookie / truly no historical data — anchor to teammate pace.
            teammates = [d for d in team_to_drivers.get(team, []) if d != driver]
            ref_hist = pd.DataFrame()
            for tm in teammates:
                tm_hist = historical_data[historical_data['Driver'] == tm].tail(5)
                if len(tm_hist) > 0:
                    ref_hist = tm_hist
                    break

            if len(ref_hist) > 0:
                avg_q1 = ref_hist['Q1_sec'].mean() * 1.005
                avg_q2 = ref_hist['Q2_sec'].mean() * 1.005
            elif len(team_hist) > 0:
                avg_q1 = team_hist['Q1_sec'].mean() * 1.005
                avg_q2 = team_hist['Q2_sec'].mean() * 1.005
            else:
                overall_q1 = historical_data['Q1_sec'].mean()
                overall_q2 = historical_data['Q2_sec'].mean()
                avg_q1 = overall_q1 * 1.015
                avg_q2 = overall_q2 * 1.015

            avg_q1_ewm = avg_q1
            avg_q2_ewm = avg_q2
            driver_q3_rate = 0.15

        # Team features: cross-year normalized + current-year isolated pace
        current_year = historical_data['Year'].max()
        team_current_year_data = historical_data[
            (historical_data['TeamName'] == team) & (historical_data['Year'] == current_year)
        ]
        if len(team_hist) > 0:
            team_q1 = team_hist['Q1_normalized'].mean() * ref_q1_mean
            team_q2 = team_hist['Q2_normalized'].mean() * ref_q2_mean
        else:
            team_q1 = avg_q1
            team_q2 = avg_q2

        if len(team_current_year_data) > 0:
            team_q1_current_year = team_current_year_data['Q1_normalized'].mean()
            team_q2_current_year = team_current_year_data['Q2_normalized'].mean()
            team_q3_current_year = team_current_year_data['Q3_sec'].dropna().mean()
        elif len(team_hist) > 0:
            team_q1_current_year = team_hist['Q1_normalized'].mean()
            team_q2_current_year = team_hist['Q2_normalized'].mean()
            team_q3_current_year = team_hist['Q3_sec'].dropna().mean()
        else:
            team_q1_current_year = avg_q1 / ref_q1_mean if ref_q1_mean > 0 else 1.0
            team_q2_current_year = avg_q2 / ref_q2_mean if ref_q2_mean > 0 else 1.0
            team_q3_current_year = circuit_q3_mean

        # --- New driver features ---
        # Use driver_hist_any for career-level signals (form trend, consistency, wet delta)
        # Use driver_hist_team for current-car teammate delta where available
        all_driver_hist = historical_data[historical_data['Driver'] == driver]

        # Form trend: slope of Q1_normalized over last 5 sessions (career-wide, not team-specific)
        recent_norm = all_driver_hist['Q1_normalized'].dropna().tail(5).values
        if len(recent_norm) >= 3:
            x = np.arange(len(recent_norm))
            form_trend = float(np.polyfit(x, recent_norm, 1)[0])
        else:
            form_trend = 0.0  # neutral — no trend signal

        # Pace consistency: std of Q1_normalized over last 5 sessions
        if len(recent_norm) >= 2:
            pace_consistency = float(np.std(recent_norm))
        else:
            # Impute with global mean consistency; SimpleImputer will handle NaN too
            pace_consistency = float(historical_data['Driver_pace_consistency'].median())
            if np.isnan(pace_consistency):
                pace_consistency = 0.01

        # Teammate Q1 delta: driver's mean Q1_norm vs teammate's mean Q1_norm
        # at the current team. Negative = faster than teammate.
        teammates = [d for d in team_to_drivers.get(team, []) if d != driver]
        teammate_norm_vals = []
        for tm in teammates:
            tm_data = historical_data[
                (historical_data['Driver'] == tm) &
                (historical_data['TeamName'] == team)
            ]['Q1_normalized'].dropna().tail(5)
            if len(tm_data) > 0:
                teammate_norm_vals.append(tm_data.mean())
        if teammate_norm_vals:
            driver_norm_vs_team = avg_q1 / ref_q1_mean if ref_q1_mean > 0 else 1.0
            teammate_q1_delta = driver_norm_vs_team - float(np.mean(teammate_norm_vals))
        else:
            teammate_q1_delta = 0.0  # no teammate data — neutral

        # Q2 form trend: slope of Q2_normalized over last 5 sessions
        recent_q2_norm = all_driver_hist['Q2_normalized'].dropna().tail(5).values
        if len(recent_q2_norm) >= 3:
            x2 = np.arange(len(recent_q2_norm))
            q2_trend = float(np.polyfit(x2, recent_q2_norm, 1)[0])
        else:
            q2_trend = 0.0

        features = {
            'Q1_sec': avg_q1,
            'Q2_sec': avg_q2,
            'Q1_Q2_diff': avg_q2 - avg_q1,
            'Q1_Q2_improvement_pct': ((avg_q1 - avg_q2) / avg_q1 * 100) if avg_q1 > 0 else 0.8,
            'Q1_normalized': avg_q1 / ref_q1_mean if ref_q1_mean > 0 else 1.0,
            'Q2_normalized': avg_q2 / ref_q2_mean if ref_q2_mean > 0 else 1.0,
            'Driver_Q1_ewm': avg_q1_ewm,
            'Driver_Q2_ewm': avg_q2_ewm,
            'Team_Q1_mean': team_q1,
            'Team_Q2_mean': team_q2,
            'Team_Q1_current_year': team_q1_current_year,
            'Team_Q2_current_year': team_q2_current_year,
            'Team_Q3_current_year': team_q3_current_year,
            'TrackType_enc': next_track_type_enc,
            'Driver_Q3_rate': driver_q3_rate,
            'Circuit_Q3_mean': circuit_q3_mean,
            'Driver_form_trend': form_trend,
            'Driver_Q2_trend': q2_trend,
            'Driver_pace_consistency': pace_consistency,
            'Teammate_Q1_delta': teammate_q1_delta,
        }

        X = pd.DataFrame([features])[FEATURE_COLS]
        X_imp = imputer.transform(X)
        X_s = scaler.transform(X_imp)

        pred_q3 = float(model.predict(X_s)[0])

        # Confidence interval: std of predictions from 20 equally-spaced tree-count slices
        n_trees = model.n_estimators
        slice_size = max(1, n_trees // 20)
        slice_preds = [
            float(model.predict(X_s, iteration_range=(0, min(t, n_trees)))[0])
            for t in range(slice_size, n_trees + slice_size, slice_size)
        ]
        confidence_std = float(np.std(slice_preds)) if len(slice_preds) > 1 else 0.0

        predictions.append({
            'Driver': driver,
            'Team': team,
            'Predicted_Q3': pred_q3,
            'Confidence_std': round(confidence_std, 3),
        })

    results_df = pd.DataFrame(predictions).sort_values('Predicted_Q3').reset_index(drop=True)
    results_df['Position'] = range(1, len(results_df) + 1)
    results_df['Predicted_Q3'] = results_df['Predicted_Q3'].astype(float)
    results_df['Position'] = results_df['Position'].astype(int)
    return results_df

# ---------------------------------------------------------------------------
# Historical accuracy tracking (improvement #13)
# ---------------------------------------------------------------------------

def load_accuracy_log():
    if ACCURACY_LOG.exists():
        with open(ACCURACY_LOG) as f:
            return json.load(f)
    return []

def save_prediction_to_log(race_info, predictions_df, model_metrics):
    log = load_accuracy_log()
    entry = {
        'race': race_info.get('name'),
        'year': race_info.get('year'),
        'round': race_info.get('round'),
        'timestamp': pd.Timestamp.now().isoformat(),
        'model_mae': model_metrics.get('mae'),
        'model_rmse': model_metrics.get('rmse'),
        'model_r2': model_metrics.get('r2'),
        'predictions': predictions_df[['Driver', 'Team', 'Position', 'Predicted_Q3']].to_dict(orient='records'),
        'actual_results': None,  # filled in by /record_actuals endpoint
    }
    log.append(entry)
    with open(ACCURACY_LOG, 'w') as f:
        json.dump(log, f, indent=2)

# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET'])
def predict():
    return render_template('loading.html')

@app.route('/predict_stream')
def predict_stream():
    def generate():
        status_updates = []

        def emit(status, message, **extra):
            payload = {'status': status, 'message': message, **extra}
            status_updates.append(payload)
            return f"data: {json.dumps(payload)}\n\n"

        try:
            yield emit('starting', 'Initializing prediction engine...')

            all_data = []
            training_years = get_training_years()
            skipped = 0
            loaded = 0

            for year in training_years:
                try:
                    schedule = fastf1.get_event_schedule(year, include_testing=False)
                    rounds = schedule['RoundNumber'].dropna().astype(int).tolist()
                except Exception:
                    rounds = list(range(1, 25))

                for round_num in rounds:
                    try:
                        session = fastf1.get_session(year, round_num, 'Q')
                        event_name = session.event['EventName']
                    except Exception:
                        event_name = f"Round {round_num}"

                    yield emit('loading', f'Loading {year} — {event_name}')

                    data = fetch_f1_data(year, round_num)
                    if data is not None and len(data) > 0:
                        all_data.append(data)
                        loaded += 1
                    else:
                        # No qualifying data for this round — season is done, stop early
                        skipped += 1
                        break

            if not all_data:
                yield emit('error', 'No data available for predictions')
                return

            yield emit('processing', f'Engineering features ({loaded} sessions loaded, {skipped} skipped)...')
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_df = engineer_features(combined_df)

            yield emit('training', 'Training XGBoost model...')

            status_ref = [None]
            def yield_status_fn(s, m):
                status_ref[0] = (s, m)

            bundle = train_model(combined_df, yield_status=yield_status_fn)
            if status_ref[0]:
                yield emit(*status_ref[0])

            next_race = get_next_race_info()
            yield emit('predicting', f"Generating predictions for {next_race['name']}...")

            predictions = predict_next_race(bundle, combined_df, next_race, next_race.get('year'))

            metrics = {
                'mae': round(bundle['mae'], 3),
                'rmse': round(bundle['rmse'], 3),
                'r2': round(bundle['r2'], 3),
            }

            save_prediction_to_log(next_race, predictions, metrics)

            predictions_list = json.loads(predictions.to_json(orient='records'))

            yield emit(
                'complete', 'Predictions ready!',
                predictions=predictions_list,
                race_info=next_race,
                metrics=metrics,
            )

        except Exception as e:
            import traceback
            print(traceback.format_exc())
            yield f"data: {json.dumps({'status': 'error', 'message': f'Error: {str(e)}'})}\n\n"

    return Response(generate(), mimetype='text/event-stream')

@app.route('/results')
def results():
    # Improvement #14 — shareable results via server-side session keyed to URL param
    result_id = request.args.get('id')
    stored = None
    if result_id:
        result_path = CACHE_DIR / f"result_{result_id}.json"
        if result_path.exists():
            with open(result_path) as f:
                stored = json.load(f)
    return render_template('results.html', stored=stored)

@app.route('/save_result', methods=['POST'])
def save_result():
    """Called by JS after receiving predictions — stores on server, returns shareable ID."""
    data = request.get_json()
    result_id = hashlib.md5(
        json.dumps(data, sort_keys=True).encode()
    ).hexdigest()[:10]
    result_path = CACHE_DIR / f"result_{result_id}.json"
    with open(result_path, 'w') as f:
        json.dump(data, f)
    return jsonify({'id': result_id})

@app.route('/history')
def history():
    """View historical prediction accuracy log."""
    log = load_accuracy_log()
    return render_template('history.html', log=log)

if __name__ == '__main__':
    app.run(debug=True)

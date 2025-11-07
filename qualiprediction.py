import fastf1
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from flask import Flask, render_template, request, jsonify, Response
import json
import time
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
fastf1.Cache.enable_cache('cache')

def fetch_f1_data(year, round_number):
    try:
        quali = fastf1.get_session(year, round_number, 'Q')
        quali.load()
        results = quali.results[['DriverNumber', 'FullName', 'TeamName', 'Q1', 'Q2', 'Q3', 'Position']]
        results = results.rename(columns={'FullName': 'Driver'})

        # Convert times to seconds
        for col in ['Q1', 'Q2', 'Q3']:
            results[col + '_sec'] = results[col].apply(
                lambda x: x.total_seconds() if pd.notnull(x) else None
            )

        # Add race metadata
        results['Year'] = year
        results['Round'] = round_number
        results['TrackName'] = quali.event['EventName']

        return results
    except Exception as e:
        print(f"Error fetching data for {year} Round {round_number}: {e}")
        return None

def convert_time_to_seconds(time_str):
    if pd.isna(time_str):
        return None
    try:
        if ':' in time_str:
            minutes, seconds = time_str.split(':')
            return float(minutes) * 60 + float(seconds)
        else:
            return float(time_str)
    except:
        return None

def engineer_features(df):
    """Create advanced features for better predictions"""
    df = df.copy()

    # Basic time features
    df['Q1_Q2_diff'] = df['Q2_sec'] - df['Q1_sec']
    df['Q1_Q2_improvement_pct'] = (df['Q1_sec'] - df['Q2_sec']) / df['Q1_sec'] * 100

    # Track-specific performance (normalization)
    for round_num in df['Round'].unique():
        round_mask = df['Round'] == round_num
        if round_mask.sum() > 0:
            q1_mean = df.loc[round_mask, 'Q1_sec'].mean()
            q2_mean = df.loc[round_mask, 'Q2_sec'].mean()
            if q1_mean > 0:
                df.loc[round_mask, 'Q1_normalized'] = df.loc[round_mask, 'Q1_sec'] / q1_mean
            if q2_mean > 0:
                df.loc[round_mask, 'Q2_normalized'] = df.loc[round_mask, 'Q2_sec'] / q2_mean

    # Driver recent form (rolling average of last 3 races)
    df = df.sort_values(['Driver', 'Year', 'Round'])
    df['Driver_Q1_rolling'] = df.groupby('Driver')['Q1_sec'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )
    df['Driver_Q2_rolling'] = df.groupby('Driver')['Q2_sec'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )

    # Team performance
    df['Team_Q1_mean'] = df.groupby(['TeamName', 'Round'])['Q1_sec'].transform('mean')
    df['Team_Q2_mean'] = df.groupby(['TeamName', 'Round'])['Q2_sec'].transform('mean')

    return df

def train_improved_model(df):
    """Train model with enhanced features and hyperparameter tuning"""

    # Define feature columns
    feature_cols = [
        'Q1_sec', 'Q2_sec', 'Q1_Q2_diff', 'Q1_Q2_improvement_pct',
        'Q1_normalized', 'Q2_normalized',
        'Driver_Q1_rolling', 'Driver_Q2_rolling',
        'Team_Q1_mean', 'Team_Q2_mean'
    ]

    # Remove rows with missing Q3 times (target variable)
    df_clean = df.dropna(subset=['Q3_sec'])

    # Prepare features and target
    X = df_clean[feature_cols].copy()
    y = df_clean['Q3_sec'].copy()

    # Handle any remaining missing values in features
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols, index=X.index)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.2, random_state=42, shuffle=True
    )

    # Scale features for better performance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Hyperparameter tuning with GridSearchCV
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 15, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    rf_model = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(
        rf_model, param_grid, cv=5, scoring='neg_mean_absolute_error',
        n_jobs=-1, verbose=0
    )

    print("Training model with hyperparameter tuning...")
    grid_search.fit(X_train_scaled, y_train)

    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")

    # Evaluate model
    y_pred_train = best_model.predict(X_train_scaled)
    y_pred_test = best_model.predict(X_test_scaled)

    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    print("\n=== Model Performance ===")
    print(f"Training MAE: {train_mae:.3f} seconds")
    print(f"Test MAE: {test_mae:.3f} seconds")
    print(f"Test RMSE: {rmse:.3f} seconds")
    print(f"Training R²: {train_r2:.3f}")
    print(f"Test R²: {test_r2:.3f}")

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n=== Feature Importance ===")
    print(feature_importance.to_string(index=False))

    return best_model, scaler, imputer, feature_cols

def get_next_race_info():
    """Get information about the next upcoming race"""
    try:
        schedule = fastf1.get_event_schedule(2025)
        today = pd.Timestamp.now()

        # Find next race
        future_races = schedule[schedule['EventDate'] > today]
        if len(future_races) > 0:
            next_race = future_races.iloc[0]
            return {
                'round': int(next_race['RoundNumber']),
                'name': str(next_race['EventName']),
                'location': str(next_race['Location']),
                'date': str(next_race['EventDate'])
            }
    except Exception as e:
        print(f"Error getting next race info: {e}")

    return {'round': None, 'name': 'Next Grand Prix', 'location': 'TBD', 'date': None}

def predict_next_race(model, scaler, imputer, feature_cols, historical_data):
    """
    Predict next race using historical performance data and trained model
    """
    driver_teams_2025 = {
        'Lando Norris': 'McLaren',
        'Oscar Piastri': 'McLaren',
        'Charles Leclerc': 'Ferrari',
        'Lewis Hamilton': 'Ferrari',
        'Max Verstappen': 'Red Bull Racing',
        'Yuki Tsunoda': 'Red Bull Racing',
        'George Russell': 'Mercedes',
        'Kimi Antonelli': 'Mercedes',
        'Fernando Alonso': 'Aston Martin',
        'Lance Stroll': 'Aston Martin',
        'Isack Hadjar': 'RB',
        'Liam Lawson': 'RB',
        'Alex Albon': 'Williams',
        'Carlos Sainz': 'Williams',
        'Nico Hulkenberg': 'Kick Sauber',
        'Gabriel Bortoleto': 'Kick Sauber',
        'Oliver Bearman': 'Haas F1 Team',
        'Esteban Ocon': 'Haas F1 Team',
        'Pierre Gasly': 'Alpine',
        'Franco Colapinto': 'Alpine'
    }

    # Get overall track average for normalization
    overall_q1_mean = historical_data['Q1_sec'].mean()
    overall_q2_mean = historical_data['Q2_sec'].mean()

    predictions = []

    for driver, team in driver_teams_2025.items():
        # Get driver's recent performance from historical data (last 5 races)
        driver_data = historical_data[historical_data['Driver'] == driver].tail(5)

        # Get team performance baseline first for fallback
        team_data = historical_data[historical_data['TeamName'] == team].tail(15)

        if len(driver_data) > 0:
            # Use recent driver averages - this represents the driver's current form
            avg_q1 = driver_data['Q1_sec'].mean()
            avg_q2 = driver_data['Q2_sec'].mean()
            avg_q1_rolling = driver_data['Driver_Q1_rolling'].iloc[-1]
            avg_q2_rolling = driver_data['Driver_Q2_rolling'].iloc[-1]
        elif len(team_data) > 0:
            # Fallback for new drivers - use team average + penalty
            avg_q1 = team_data['Q1_sec'].mean() * 1.002  # Slightly slower than team average
            avg_q2 = team_data['Q2_sec'].mean() * 1.002
            avg_q1_rolling = avg_q1
            avg_q2_rolling = avg_q2
        else:
            # Ultimate fallback - backmarker pace
            avg_q1 = overall_q1_mean * 1.015  # Much slower than average
            avg_q2 = overall_q2_mean * 1.015
            avg_q1_rolling = avg_q1
            avg_q2_rolling = avg_q2

        # Get team performance baseline
        if len(team_data) > 0:
            team_q1_mean = team_data['Q1_sec'].mean()
            team_q2_mean = team_data['Q2_sec'].mean()
        else:
            team_q1_mean = avg_q1
            team_q2_mean = avg_q2

        # Create feature vector with realistic values
        features = {
            'Q1_sec': avg_q1,
            'Q2_sec': avg_q2,
            'Q1_Q2_diff': avg_q2 - avg_q1,
            'Q1_Q2_improvement_pct': ((avg_q1 - avg_q2) / avg_q1 * 100) if avg_q1 > 0 else 0.8,
            'Q1_normalized': avg_q1 / overall_q1_mean if overall_q1_mean > 0 else 1.0,
            'Q2_normalized': avg_q2 / overall_q2_mean if overall_q2_mean > 0 else 1.0,
            'Driver_Q1_rolling': avg_q1_rolling,
            'Driver_Q2_rolling': avg_q2_rolling,
            'Team_Q1_mean': team_q1_mean,
            'Team_Q2_mean': team_q2_mean
        }

        # Create DataFrame for prediction
        X_pred = pd.DataFrame([features])[feature_cols]
        X_pred_imputed = imputer.transform(X_pred)
        X_pred_scaled = scaler.transform(X_pred_imputed)

        # Predict Q3 time
        predicted_q3 = model.predict(X_pred_scaled)[0]

        predictions.append({
            'Driver': driver,
            'Team': team,
            'Predicted_Q3': predicted_q3
        })

    results_df = pd.DataFrame(predictions)
    results_df = results_df.sort_values('Predicted_Q3').reset_index(drop=True)
    results_df['Position'] = range(1, len(results_df) + 1)

    # Convert to native Python types for JSON serialization
    results_df['Predicted_Q3'] = results_df['Predicted_Q3'].astype(float)
    results_df['Position'] = results_df['Position'].astype(int)

    return results_df

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_stream')
def predict_stream():
    """Stream prediction progress to the frontend"""
    def generate():
        try:
            yield f"data: {json.dumps({'status': 'starting', 'message': 'Initializing prediction engine...'})}\n\n"
            time.sleep(0.5)

            # Fetch data from 2024 and 2025 for better training
            all_data = []
            for year in [2024, 2025]:
                for round_num in range(1, 25):
                    # Get event name
                    try:
                        session = fastf1.get_session(year, round_num, 'Q')
                        event_name = session.event['EventName']
                    except:
                        event_name = f"Round {round_num}"

                    loading_message = f'Loading {year} - {event_name}'
                    yield f"data: {json.dumps({'status': 'loading', 'message': loading_message})}\n\n"

                    data = fetch_f1_data(year, round_num)
                    if data is not None and len(data) > 0:
                        all_data.append(data)

            if not all_data:
                yield f"data: {json.dumps({'status': 'error', 'message': 'No data available for predictions'})}\n\n"
                return

            yield f"data: {json.dumps({'status': 'processing', 'message': 'Engineering features...'})}\n\n"

            # Combine all data
            combined_df = pd.concat(all_data, ignore_index=True)

            # Engineer features
            combined_df = engineer_features(combined_df)

            yield f"data: {json.dumps({'status': 'training', 'message': 'Training ML model with hyperparameter tuning...'})}\n\n"

            # Train improved model
            model, scaler, imputer, feature_cols = train_improved_model(combined_df)

            # Get next race info
            next_race = get_next_race_info()
            race_message = f"Generating predictions for {next_race['name']}..."
            yield f"data: {json.dumps({'status': 'predicting', 'message': race_message})}\n\n"

            # Make predictions for next race
            predictions = predict_next_race(model, scaler, imputer, feature_cols, combined_df)

            # Store in session or return with results
            predictions_json = predictions.to_json(orient='records')

            yield f"data: {json.dumps({'status': 'complete', 'message': 'Predictions ready!', 'predictions': json.loads(predictions_json), 'race_info': next_race})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'status': 'error', 'message': f'Error: {str(e)}'})}\n\n"

    return Response(generate(), mimetype='text/event-stream')

@app.route('/predict', methods=['GET'])
def predict():
    """Show loading page that streams progress"""
    return render_template('loading.html')

@app.route('/results')
def results():
    """Show results page"""
    # This will be called by JavaScript after predictions are complete
    return render_template('results.html')

if __name__ == '__main__':
    app.run(debug=True)

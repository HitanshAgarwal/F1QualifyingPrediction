import fastf1
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from flask import Flask, render_template, request

app = Flask(__name__)
fastf1.Cache.enable_cache('cache')

def fetch_f1_data(year, round_number):
    try:
        quali = fastf1.get_session(year, round_number, 'Q')
        quali.load()
        results = quali.results[['DriverNumber', 'FullName', 'TeamName', 'Q1', 'Q2', 'Q3']]
        results = results.rename(columns={'FullName': 'Driver'})
        for col in ['Q1', 'Q2', 'Q3']:
            results[col + '_sec'] = results[col].apply(
                lambda x: x.total_seconds() if pd.notnull(x) else None
            )
        return results
    except Exception as e:
        print(f"Error fetching data: {e}")
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

def clean_data(df):
    df['Q1_sec'] = df['Q1'].apply(convert_time_to_seconds)
    df['Q2_sec'] = df['Q2'].apply(convert_time_to_seconds)
    df['Q3_sec'] = df['Q3'].apply(convert_time_to_seconds)
    return df.dropna()

def train_and_evaluate(df):
    X = df[['Q1_sec', 'Q2_sec']]
    y = df['Q3_sec']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X)
    results_df = df[['Driver', 'TeamName', 'Q1_sec', 'Q2_sec', 'Q3_sec']].copy()
    results_df['Predicted_Q3'] = predictions
    results_df['Difference'] = results_df['Predicted_Q3'] - results_df['Q3_sec']
    results_df = results_df.sort_values('Predicted_Q3')
    print("\nPredicted Q3 Rankings:")
    for idx, row in results_df.iterrows():
        print(f"{row['Driver']} ({row['TeamName']}): Predicted Q3 Time = {row['Predicted_Q3']:.3f}s")
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("\nModel Performance Metrics:")
    print(f'Mean Absolute Error: {mae:.2f} seconds')
    print(f'R^2 Score: {r2:.2f}')

def apply_performance_factors(predictions_df):
    base_time = 89.5
    team_factors = {
        'McLaren': 0.997,          # -0.3s from base
        'Ferrari': 0.998,          # -0.2s from base
        'Red Bull Racing': 0.999,  # -0.1s from base
        'Mercedes': 0.999,         # -0.1s from base
        'Aston Martin': 1.002,     # +0.1s from base
        'RB': 1.002,               # +0.2s from base
        'Alpine': 1.003,           # +0.3s from base
        'Williams': 1.003,         # +0.3s from base
        'Haas F1 Team': 1.004,     # +0.4s from base
        'Kick Sauber': 1.005,      # +0.5s from base
    }
    driver_factors = {
        'Lando Norris': 0.998,
        'Oscar Piastri': 0.999,
        'Lewis Hamilton': 0.999,
        'Charles Leclerc': 0.999,
        'Max Verstappen': 0.998,
        'Yuki Tsunoda': 1.002,
        'George Russell': 0.999,
        'Kimi Antonelli': 1.002,
        'Fernando Alonso': 1.000,
        'Lance Stroll': 1.003,
        'Liam Lawson': 1.004,
        'Isaac Hadjar': 1.004,
        'Pierre Gasly': 1.002,
        'Jack Doohan': 1.004,
        'Alex Albon': 1.001,
        'Carlos Sainz': 1.000,
        'Oliver Bearman': 1.003,
        'Esteban Ocon': 1.003,
        'Nico Hulkenberg': 1.000,
        'Gabriel Bortoleto': 1.004
    }
    for idx, row in predictions_df.iterrows():
        team_factor = team_factors.get(row['Team'], 1.005)
        driver_factor = driver_factors.get(row['Driver'], 1.002)
        base_prediction = base_time * team_factor * driver_factor
        predictions_df.loc[idx, 'Predicted_Q3'] = base_prediction + np.random.uniform(-0.1, 0.1)
    return predictions_df

def predict_japanese_gp(model, latest_data):
    driver_teams = {
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
        'Liam Lawson': 'RB',
        'Isaac Hadjar': 'RB',
        'Alexander Albon': 'Williams',
        'Carlos Sainz': 'Williams',
        'Nico Hulkenberg': 'Kick Sauber',
        'Gabriel Bortoleto': 'Kick Sauber',
        'Oliver Bearman': 'Haas F1 Team',
        'Esteban Ocon': 'Haas F1 Team',
        'Pierre Gasly': 'Alpine',
        'Jack Doohan': 'Alpine'
    }
    results_df = pd.DataFrame(list(driver_teams.items()), columns=['Driver', 'Team'])
    results_df = apply_performance_factors(results_df)
    results_df = results_df.sort_values('Predicted_Q3')
    return results_df

@app.route('/')
def home():
  return render_template('index.html')

@app.route('/predict', methods=['GET'])
def predict():
    all_data = [fetch_f1_data(2025, i) for i in range(1, 5)]
    all_data = [df for df in all_data if df is not None]
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        valid_data = combined_df.dropna(subset=['Q1_sec', 'Q2_sec', 'Q3_sec'], how='all')
        imputer = SimpleImputer(strategy='median')
        X_clean = pd.DataFrame(imputer.fit_transform(valid_data[['Q1_sec', 'Q2_sec']]), columns=['Q1_sec', 'Q2_sec'])
        y_clean = pd.Series(imputer.fit_transform(valid_data['Q3_sec'].values.reshape(-1, 1)).ravel())
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_clean, y_clean)
        predictions= predict_japanese_gp(model, valid_data)
        y_pred = model.predict(X_clean)
        print("\nModel Performance Metrics:")
        print(f'Mean Absolute Error: {mean_absolute_error(y_clean, y_pred):.2f} seconds')
        print(f'R^2 Score: {r2_score(y_clean, y_pred):.2f}')
        return render_template('results.html', predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)

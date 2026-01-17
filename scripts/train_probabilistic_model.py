"""
Train probabilistic model to predict P(peak_temp > threshold).
Uses quantile regression to estimate full temperature distribution.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import xgboost as xgb
from sklearn.model_selection import train_test_split
import pickle

def prepare_features_with_forecast(hourly_file='data/raw/klga_hourly_full_history.csv'):
    """Prepare features from hourly data."""
    hourly_df = pd.read_csv(hourly_file)
    hourly_df['valid'] = pd.to_datetime(hourly_df['valid'])
    hourly_df['date'] = hourly_df['valid'].dt.date
    
    # Calculate daily peak temperature (target)
    daily_peaks = hourly_df.groupby('date')['tmpf'].max().reset_index()
    daily_peaks.columns = ['date', 'peak_temp']
    
    # Aggregate hourly data
    daily_stats = hourly_df.groupby('date').agg({
        'tmpf': ['mean', 'min', 'max', 'std'],
        'dwpf': ['mean', 'min', 'max'],
        'relh': ['mean', 'max'],
        'sknt': ['mean', 'max'],
        'vsby': ['mean', 'min']
    }).reset_index()
    
    daily_stats.columns = ['date', 
        'temp_mean', 'temp_min', 'temp_max', 'temp_std',
        'dewpoint_mean', 'dewpoint_min', 'dewpoint_max',
        'humidity_mean', 'humidity_max',
        'wind_mean', 'wind_max',
        'visibility_mean', 'visibility_min'
    ]
    
    df = daily_peaks.merge(daily_stats, on='date')
    df['date'] = pd.to_datetime(df['date'])
    
    # Time features
    df['month'] = df['date'].dt.month
    df['day_of_year'] = df['date'].dt.dayofyear
    df['day_of_week'] = df['date'].dt.dayofweek
    
    # Cyclical encoding
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    
    # Lag features
    for col in ['temp_mean', 'temp_max', 'dewpoint_mean', 'humidity_mean']:
        df[f'{col}_lag1'] = df[col].shift(1)
        df[f'{col}_lag2'] = df[col].shift(2)
    
    df['peak_temp_lag1'] = df['peak_temp'].shift(1)
    df['peak_temp_lag2'] = df['peak_temp'].shift(2)
    df['peak_temp_lag7'] = df['peak_temp'].shift(7)
    
    # Rolling features
    df['peak_temp_roll7'] = df['peak_temp'].shift(1).rolling(window=7, min_periods=1).mean()
    df['peak_temp_roll7_std'] = df['peak_temp'].shift(1).rolling(window=7, min_periods=1).std()
    
    return df.dropna()

def train_quantile_models(data_file='data/raw/klga_hourly_full_history.csv'):
    """Train models for different quantiles to estimate distribution."""
    
    print("=" * 70)
    print("Training Probabilistic Temperature Model")
    print("=" * 70)
    
    print("\nPreparing features...")
    df = prepare_features_with_forecast(data_file)
    print(f"✓ {len(df):,} records prepared")
    
    feature_cols = [
        'month', 'day_of_year', 'day_of_week',
        'month_sin', 'month_cos', 'day_of_year_sin', 'day_of_year_cos',
        'temp_mean_lag1', 'temp_mean_lag2',
        'temp_max_lag1', 'temp_max_lag2',
        'dewpoint_mean_lag1', 'dewpoint_mean_lag2',
        'humidity_mean_lag1', 'humidity_mean_lag2',
        'peak_temp_lag1', 'peak_temp_lag2', 'peak_temp_lag7',
        'peak_temp_roll7', 'peak_temp_roll7_std'
    ]
    
    X = df[feature_cols]
    y = df['peak_temp']
    dates = df['date']
    
    # Temporal split
    train_mask = dates < '2021-01-01'
    val_mask = (dates >= '2021-01-01') & (dates < '2023-01-01')
    test_mask = dates >= '2023-01-01'
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    print(f"\nTemporal Split:")
    print(f"  Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    
    # Train models for different quantiles
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    models = {}
    
    print(f"\nTraining quantile models...")
    for q in quantiles:
        print(f"  Training quantile {q}...", end=' ')
        
        model = xgb.XGBRegressor(
            objective='reg:quantileerror',
            quantile_alpha=q,
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train, verbose=False)
        models[q] = model
        print("✓")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    predictions = {}
    for q, model in models.items():
        predictions[q] = model.predict(X_test)
    
    # Calculate coverage (how often true value falls in predicted interval)
    coverage_50 = np.mean((y_test >= predictions[0.25]) & (y_test <= predictions[0.75]))
    coverage_80 = np.mean((y_test >= predictions[0.1]) & (y_test <= predictions[0.9]))
    
    print(f"\nPrediction Interval Coverage:")
    print(f"  50% interval (Q25-Q75): {coverage_50*100:.1f}% (target: 50%)")
    print(f"  80% interval (Q10-Q90): {coverage_80*100:.1f}% (target: 80%)")
    
    # Show example predictions
    print(f"\nExample Predictions (first 5 test days):")
    for i in range(min(5, len(y_test))):
        actual = y_test.iloc[i]
        print(f"  Day {i+1}: Actual={actual:.1f}°F | "
              f"Q10={predictions[0.1][i]:.1f} | "
              f"Q50={predictions[0.5][i]:.1f} | "
              f"Q90={predictions[0.9][i]:.1f}")
    
    # Save models
    model_file = 'models/probabilistic_temp_model.pkl'
    print(f"\nSaving models to {model_file}...")
    with open(model_file, 'wb') as f:
        pickle.dump({
            'models': models,
            'feature_cols': feature_cols,
            'quantiles': quantiles,
            'train_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'coverage': {
                '50%': coverage_50,
                '80%': coverage_80
            }
        }, f)
    print("✓ Models saved")
    
    return models, feature_cols

if __name__ == "__main__":
    models, features = train_quantile_models()
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print("\nUse these models to estimate P(temp > threshold) for betting.")

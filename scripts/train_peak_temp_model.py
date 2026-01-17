"""
Train XGBoost model to predict daily peak temperature at KLGA.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

def prepare_features(df):
    """Add time-based features for prediction."""
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # Time features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_year'] = df['date'].dt.dayofyear
    df['day_of_week'] = df['date'].dt.dayofweek
    df['week_of_year'] = df['date'].dt.isocalendar().week
    
    # Cyclical encoding for seasonal patterns
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    
    # Lag features (previous days' temperatures)
    df['peak_temp_lag1'] = df['peak_temp'].shift(1)
    df['peak_temp_lag2'] = df['peak_temp'].shift(2)
    df['peak_temp_lag3'] = df['peak_temp'].shift(3)
    df['peak_temp_lag7'] = df['peak_temp'].shift(7)
    
    # Rolling averages (using past data only)
    df['peak_temp_roll7'] = df['peak_temp'].shift(1).rolling(window=7, min_periods=1).mean()
    df['peak_temp_roll30'] = df['peak_temp'].shift(1).rolling(window=30, min_periods=1).mean()
    
    # Lag features for other weather variables (previous day)
    df['dewpoint_lag1'] = df['avg_dewpoint'].shift(1)
    df['humidity_lag1'] = df['avg_humidity'].shift(1)
    df['wind_lag1'] = df['avg_wind'].shift(1)
    df['visibility_lag1'] = df['avg_visibility'].shift(1)
    
    return df

def train_model(data_file='data/raw/klga_daily_peaks.csv'):
    """Train XGBoost model to predict peak temperature."""
    
    print("=" * 70)
    print("Training Peak Temperature Prediction Model")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    df = pd.read_csv(data_file)
    print(f"✓ Loaded {len(df):,} daily records")
    
    # Prepare features
    print("\nPreparing features...")
    df = prepare_features(df)
    
    # Drop rows with NaN (from lag features)
    df = df.dropna()
    print(f"✓ {len(df):,} records after feature engineering")
    
    # Define features and target (only using past data!)
    feature_cols = [
        'month', 'day_of_year', 'day_of_week', 'week_of_year',
        'month_sin', 'month_cos', 'day_of_year_sin', 'day_of_year_cos',
        'peak_temp_lag1', 'peak_temp_lag2', 'peak_temp_lag3', 'peak_temp_lag7',
        'peak_temp_roll7', 'peak_temp_roll30',
        'dewpoint_lag1', 'humidity_lag1', 'wind_lag1', 'visibility_lag1'
    ]
    
    X = df[feature_cols]
    y = df['peak_temp']
    
    # Split data (80/20 train/test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False  # Don't shuffle time series
    )
    
    print(f"\nTrain set: {len(X_train):,} records")
    print(f"Test set: {len(X_test):,} records")
    
    # Train XGBoost model
    print("\nTraining XGBoost model...")
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train, verbose=False)
    print("✓ Model trained")
    
    # Evaluate
    print("\nEvaluating model...")
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    print("\nModel Performance:")
    print(f"  Train MAE:  {train_mae:.2f}°F")
    print(f"  Test MAE:   {test_mae:.2f}°F")
    print(f"  Train RMSE: {train_rmse:.2f}°F")
    print(f"  Test RMSE:  {test_rmse:.2f}°F")
    print(f"  Train R²:   {train_r2:.4f}")
    print(f"  Test R²:    {test_r2:.4f}")
    
    # Feature importance
    print("\nTop 10 Most Important Features:")
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for idx, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']:25s} {row['importance']:.4f}")
    
    # Save model
    model_file = 'models/peak_temp_xgboost.pkl'
    print(f"\nSaving model to {model_file}...")
    with open(model_file, 'wb') as f:
        pickle.dump({
            'model': model,
            'feature_cols': feature_cols,
            'train_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'metrics': {
                'test_mae': test_mae,
                'test_rmse': test_rmse,
                'test_r2': test_r2
            }
        }, f)
    print("✓ Model saved")
    
    return model, feature_cols

if __name__ == "__main__":
    model, features = train_model()
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print("\nModel saved to: models/peak_temp_xgboost.pkl")
    print("Use this model to predict daily peak temperatures.")

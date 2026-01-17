"""
Train XGBoost model to predict daily peak temperature using hourly historical data.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

def prepare_daily_features(hourly_df):
    """Aggregate hourly data into daily features for prediction."""
    hourly_df = hourly_df.copy()
    hourly_df['valid'] = pd.to_datetime(hourly_df['valid'])
    hourly_df['date'] = hourly_df['valid'].dt.date
    
    # Calculate daily peak temperature (target)
    daily_peaks = hourly_df.groupby('date')['tmpf'].max().reset_index()
    daily_peaks.columns = ['date', 'peak_temp']
    
    # Aggregate hourly data into daily statistics
    daily_stats = hourly_df.groupby('date').agg({
        'tmpf': ['mean', 'min', 'max', 'std'],
        'dwpf': ['mean', 'min', 'max'],
        'relh': ['mean', 'max'],
        'sknt': ['mean', 'max'],
        'gust': ['max', 'mean'],
        'vsby': ['mean', 'min'],
        'drct': ['mean']
    }).reset_index()
    
    # Flatten column names
    daily_stats.columns = ['date', 
        'temp_mean', 'temp_min', 'temp_max', 'temp_std',
        'dewpoint_mean', 'dewpoint_min', 'dewpoint_max',
        'humidity_mean', 'humidity_max',
        'wind_mean', 'wind_max',
        'gust_max', 'gust_mean',
        'visibility_mean', 'visibility_min',
        'wind_dir_mean'
    ]
    
    # Merge with peaks
    df = daily_peaks.merge(daily_stats, on='date')
    df['date'] = pd.to_datetime(df['date'])
    
    # Time features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day_of_year'] = df['date'].dt.dayofyear
    df['day_of_week'] = df['date'].dt.dayofweek
    df['week_of_year'] = df['date'].dt.isocalendar().week
    
    # Cyclical encoding
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    
    # Lag features - previous days (shift by 1 to avoid data leakage)
    for col in ['temp_mean', 'temp_min', 'temp_max', 'dewpoint_mean', 'humidity_mean', 'wind_mean']:
        df[f'{col}_lag1'] = df[col].shift(1)
        df[f'{col}_lag2'] = df[col].shift(2)
        df[f'{col}_lag3'] = df[col].shift(3)
    
    # Peak temp lags
    df['peak_temp_lag1'] = df['peak_temp'].shift(1)
    df['peak_temp_lag2'] = df['peak_temp'].shift(2)
    df['peak_temp_lag3'] = df['peak_temp'].shift(3)
    df['peak_temp_lag7'] = df['peak_temp'].shift(7)
    
    # Rolling averages (using past data only)
    df['peak_temp_roll7'] = df['peak_temp'].shift(1).rolling(window=7, min_periods=1).mean()
    df['peak_temp_roll14'] = df['peak_temp'].shift(1).rolling(window=14, min_periods=1).mean()
    df['peak_temp_roll30'] = df['peak_temp'].shift(1).rolling(window=30, min_periods=1).mean()
    
    df['temp_mean_roll7'] = df['temp_mean'].shift(1).rolling(window=7, min_periods=1).mean()
    df['dewpoint_mean_roll7'] = df['dewpoint_mean'].shift(1).rolling(window=7, min_periods=1).mean()
    
    return df

def train_model(hourly_file='data/raw/klga_hourly_full_history.csv'):
    """Train XGBoost model using hourly historical data."""
    
    print("=" * 70)
    print("Training Peak Temperature Prediction Model (Hourly Data)")
    print("=" * 70)
    
    # Load hourly data
    print("\nLoading hourly data...")
    hourly_df = pd.read_csv(hourly_file)
    print(f"✓ Loaded {len(hourly_df):,} hourly records")
    
    # Prepare daily features
    print("\nAggregating to daily features...")
    df = prepare_daily_features(hourly_df)
    print(f"✓ Created {len(df):,} daily records")
    
    # Drop rows with NaN (from lag features)
    df = df.dropna()
    print(f"✓ {len(df):,} records after feature engineering")
    
    # Define features (only using past data!)
    feature_cols = [
        'month', 'day_of_year', 'day_of_week', 'week_of_year',
        'month_sin', 'month_cos', 'day_of_year_sin', 'day_of_year_cos',
        'temp_mean_lag1', 'temp_mean_lag2', 'temp_mean_lag3',
        'temp_min_lag1', 'temp_min_lag2', 'temp_min_lag3',
        'temp_max_lag1', 'temp_max_lag2', 'temp_max_lag3',
        'dewpoint_mean_lag1', 'dewpoint_mean_lag2', 'dewpoint_mean_lag3',
        'humidity_mean_lag1', 'humidity_mean_lag2', 'humidity_mean_lag3',
        'wind_mean_lag1', 'wind_mean_lag2', 'wind_mean_lag3',
        'peak_temp_lag1', 'peak_temp_lag2', 'peak_temp_lag3', 'peak_temp_lag7',
        'peak_temp_roll7', 'peak_temp_roll14', 'peak_temp_roll30',
        'temp_mean_roll7', 'dewpoint_mean_roll7'
    ]
    
    X = df[feature_cols]
    y = df['peak_temp']
    dates = df['date']
    
    # Temporal split: train/val/test
    train_mask = dates < '2021-01-01'
    val_mask = (dates >= '2021-01-01') & (dates < '2023-01-01')
    test_mask = dates >= '2023-01-01'
    
    X_train = X[train_mask]
    X_val = X[val_mask]
    X_test = X[test_mask]
    y_train = y[train_mask]
    y_val = y[val_mask]
    y_test = y[test_mask]
    
    print(f"\nTemporal Split:")
    print(f"  Train: before 2021 ({len(X_train):,} records)")
    print(f"  Val:   2021-2022 ({len(X_val):,} records)")
    print(f"  Test:  2023+ ({len(X_test):,} records)")
    
    # Train XGBoost model with early stopping on validation set
    print("\nTraining XGBoost model with validation...")
    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False
    )
    print(f"✓ Model trained (stopped at {model.best_iteration} iterations)")
    
    # Evaluate on all sets
    print("\nEvaluating model...")
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)
    
    train_mae = mean_absolute_error(y_train, y_pred_train)
    val_mae = mean_absolute_error(y_val, y_pred_val)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_r2 = r2_score(y_train, y_pred_train)
    val_r2 = r2_score(y_val, y_pred_val)
    test_r2 = r2_score(y_test, y_pred_test)
    
    print("\nModel Performance:")
    print(f"  Train MAE:  {train_mae:.2f}°F  |  RMSE: {train_rmse:.2f}°F  |  R²: {train_r2:.4f}")
    print(f"  Val MAE:    {val_mae:.2f}°F  |  RMSE: {val_rmse:.2f}°F  |  R²: {val_r2:.4f}")
    print(f"  Test MAE:   {test_mae:.2f}°F  |  RMSE: {test_rmse:.2f}°F  |  R²: {test_r2:.4f}")
    
    # Feature importance
    print("\nTop 15 Most Important Features:")
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for idx, row in feature_importance.head(15).iterrows():
        print(f"  {row['feature']:30s} {row['importance']:.4f}")
    
    # Save model
    model_file = 'models/peak_temp_hourly_xgboost.pkl'
    print(f"\nSaving model to {model_file}...")
    with open(model_file, 'wb') as f:
        pickle.dump({
            'model': model,
            'feature_cols': feature_cols,
            'train_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'metrics': {
                'train_mae': train_mae,
                'val_mae': val_mae,
                'test_mae': test_mae,
                'test_rmse': test_rmse,
                'test_r2': test_r2
            },
            'best_iteration': model.best_iteration
        }, f)
    print("✓ Model saved")
    
    return model, feature_cols

if __name__ == "__main__":
    model, features = train_model()
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print("\nModel saved to: models/peak_temp_hourly_xgboost.pkl")
    print("This model predicts tomorrow's peak temp using today's hourly data.")

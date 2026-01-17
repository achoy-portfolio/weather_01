"""
Train model with intraday features - uses temperature trends from earlier in the day
to predict the day's peak temperature.
"""

import pandas as pd
import numpy as np
from datetime import datetime, time
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

def prepare_intraday_features(hourly_file='data/raw/klga_hourly_full_history.csv'):
    """
    Prepare features including intraday temperature data.
    For each day, calculate what the temp was at various times (6am, 9am, noon, 3pm)
    and use that to predict the day's peak.
    """
    print("Loading hourly data...")
    df = pd.read_csv(hourly_file)
    df['valid'] = pd.to_datetime(df['valid'])
    df['date'] = df['valid'].dt.date
    df['hour'] = df['valid'].dt.hour
    
    print(f"✓ Loaded {len(df):,} hourly records")
    
    # Calculate daily peak (target)
    daily_peaks = df.groupby('date')['tmpf'].max().reset_index()
    daily_peaks.columns = ['date', 'peak_temp']
    
    # Get temperature at specific times of day
    print("Extracting intraday temperatures...")
    
    # Morning temps (6 AM, 9 AM)
    temp_6am = df[df['hour'] == 6].groupby('date')['tmpf'].first().reset_index()
    temp_6am.columns = ['date', 'temp_6am']
    
    temp_9am = df[df['hour'] == 9].groupby('date')['tmpf'].first().reset_index()
    temp_9am.columns = ['date', 'temp_9am']
    
    # Midday temps (noon, 3 PM)
    temp_noon = df[df['hour'] == 12].groupby('date')['tmpf'].first().reset_index()
    temp_noon.columns = ['date', 'temp_noon']
    
    temp_3pm = df[df['hour'] == 15].groupby('date')['tmpf'].first().reset_index()
    temp_3pm.columns = ['date', 'temp_3pm']
    
    # Daily statistics
    daily_stats = df.groupby('date').agg({
        'tmpf': ['mean', 'min', 'max', 'std'],
        'dwpf': ['mean'],
        'relh': ['mean'],
        'sknt': ['mean']
    }).reset_index()
    
    daily_stats.columns = ['date', 'temp_mean', 'temp_min', 'temp_max', 'temp_std',
                           'dewpoint_mean', 'humidity_mean', 'wind_mean']
    
    # Merge all features
    daily = daily_peaks.copy()
    daily = daily.merge(daily_stats, on='date', how='left')
    daily = daily.merge(temp_6am, on='date', how='left')
    daily = daily.merge(temp_9am, on='date', how='left')
    daily = daily.merge(temp_noon, on='date', how='left')
    daily = daily.merge(temp_3pm, on='date', how='left')
    
    daily['date'] = pd.to_datetime(daily['date'])
    
    # Time features
    daily['month'] = daily['date'].dt.month
    daily['day_of_year'] = daily['date'].dt.dayofyear
    daily['day_of_week'] = daily['date'].dt.dayofweek
    
    # Cyclical encoding
    daily['month_sin'] = np.sin(2 * np.pi * daily['month'] / 12)
    daily['month_cos'] = np.cos(2 * np.pi * daily['month'] / 12)
    daily['day_of_year_sin'] = np.sin(2 * np.pi * daily['day_of_year'] / 365)
    daily['day_of_year_cos'] = np.cos(2 * np.pi * daily['day_of_year'] / 365)
    
    # Lag features from previous days
    for col in ['temp_mean', 'temp_max', 'temp_min', 'dewpoint_mean']:
        daily[f'{col}_lag1'] = daily[col].shift(1)
        daily[f'{col}_lag2'] = daily[col].shift(2)
    
    daily['peak_temp_lag1'] = daily['peak_temp'].shift(1)
    daily['peak_temp_lag2'] = daily['peak_temp'].shift(2)
    daily['peak_temp_lag7'] = daily['peak_temp'].shift(7)
    
    # Rolling features
    daily['peak_temp_roll7'] = daily['peak_temp'].shift(1).rolling(window=7, min_periods=1).mean()
    daily['peak_temp_roll7_std'] = daily['peak_temp'].shift(1).rolling(window=7, min_periods=1).std()
    
    # Temperature change features (warming/cooling trends)
    daily['temp_change_6am_to_9am'] = daily['temp_9am'] - daily['temp_6am']
    daily['temp_change_9am_to_noon'] = daily['temp_noon'] - daily['temp_9am']
    daily['temp_change_noon_to_3pm'] = daily['temp_3pm'] - daily['temp_noon']
    
    # High so far by different times
    daily['high_by_9am'] = daily[['temp_6am', 'temp_9am']].max(axis=1)
    daily['high_by_noon'] = daily[['temp_6am', 'temp_9am', 'temp_noon']].max(axis=1)
    daily['high_by_3pm'] = daily[['temp_6am', 'temp_9am', 'temp_noon', 'temp_3pm']].max(axis=1)
    
    return daily.dropna()

def train_model(hourly_file='data/raw/klga_hourly_full_history.csv'):
    """Train model with intraday features."""
    
    print("=" * 70)
    print("Training Peak Temperature Model with Intraday Features")
    print("=" * 70)
    
    df = prepare_intraday_features(hourly_file)
    print(f"✓ Prepared {len(df):,} daily records with intraday features")
    
    # Define features
    feature_cols = [
        # Time features
        'month', 'day_of_year', 'day_of_week',
        'month_sin', 'month_cos', 'day_of_year_sin', 'day_of_year_cos',
        
        # Previous days
        'peak_temp_lag1', 'peak_temp_lag2', 'peak_temp_lag7',
        'temp_mean_lag1', 'temp_max_lag1', 'temp_min_lag1',
        'dewpoint_mean_lag1', 'dewpoint_mean_lag2',
        
        # Rolling averages
        'peak_temp_roll7', 'peak_temp_roll7_std',
        
        # TODAY'S intraday temps (this is the key addition!)
        'temp_6am', 'temp_9am', 'temp_noon', 'temp_3pm',
        'high_by_9am', 'high_by_noon', 'high_by_3pm',
        'temp_change_6am_to_9am', 'temp_change_9am_to_noon', 'temp_change_noon_to_3pm'
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
    
    # Train model
    print("\nTraining XGBoost with early stopping...")
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
    
    # Evaluate
    print("\nEvaluating...")
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
    model_file = 'models/peak_temp_intraday_xgboost.pkl'
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
    print("\nThis model uses intraday temperature data to improve predictions.")
    print("When predicting, provide today's temps at 6am, 9am, noon, 3pm.")

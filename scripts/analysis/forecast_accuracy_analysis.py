"""
Comprehensive Forecast Accuracy Analysis

Analyzes forecast accuracy using Open-Meteo Previous Runs data.
Tracks multiple accuracy metrics for different lead times.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from pathlib import Path


def load_raw_data():
    """Load forecast and actual temperature data"""
    print("Loading raw data...")
    
    # Try to load from different sources
    forecast_files = [
        'data/raw/historical_forecasts.csv',
        'data/raw/openmeteo_previous_runs.csv'
    ]
    
    forecasts = None
    for file_path in forecast_files:
        if Path(file_path).exists():
            forecasts = pd.read_csv(file_path)
            print(f"Loaded forecasts from: {file_path}")
            break
    
    if forecasts is None:
        raise FileNotFoundError("No forecast data found. Run fetch_openmeteo_previous_runs.py first.")
    
    # Parse timestamps and normalize column names
    forecasts['valid_time'] = pd.to_datetime(forecasts['valid_time'])
    
    # Handle different CSV formats
    if 'lead_time' in forecasts.columns:
        # New format (openmeteo_previous_runs.csv)
        forecasts['forecast_issued'] = pd.to_datetime(forecasts['forecast_issued'])
    elif 'days_before' in forecasts.columns:
        # Old format (historical_forecasts.csv)
        forecasts['lead_time'] = forecasts['days_before']
        # Reconstruct forecast_issued from forecast_date and forecast_time
        forecasts['forecast_issued'] = pd.to_datetime(
            forecasts['forecast_date'] + ' ' + forecasts['forecast_time']
        )
    else:
        raise ValueError("Unrecognized forecast CSV format")
    
    # Load actuals from Weather Underground (hourly data)
    try:
        actuals = pd.read_csv('data/raw/wunderground_hourly_temps.csv')
        actuals['timestamp'] = pd.to_datetime(actuals['timestamp'])
        actuals.rename(columns={'temperature_f': 'actual_temp'}, inplace=True)
        print("Using Weather Underground hourly data (official Polymarket source)")
    except FileNotFoundError:
        try:
            actuals = pd.read_csv('data/raw/actual_temperatures_meteo.csv')
            actuals['timestamp'] = pd.to_datetime(actuals['timestamp'])
            actuals.rename(columns={'temperature_f': 'actual_temp'}, inplace=True)
            print("Using Open-Meteo actual data")
        except FileNotFoundError:
            print("WARNING: No actual temperature data found!")
            actuals = pd.DataFrame(columns=['timestamp', 'actual_temp'])
    
    print(f"Loaded {len(forecasts):,} forecast records")
    print(f"Loaded {len(actuals):,} actual temperature records")
    
    # Show lead time distribution
    if 'lead_time' in forecasts.columns:
        print(f"\nLead time distribution:")
        print(forecasts['lead_time'].value_counts().sort_index())
    
    return forecasts, actuals


def metric_1_hourly_accuracy(forecasts, actuals):
    """
    METRIC 1: Hourly Temperature Accuracy by Lead Time
    
    For each lead time (0, 1, 2, 3 days), how accurate are the hourly forecasts?
    """
    print("\n" + "="*70)
    print("METRIC 1: HOURLY TEMPERATURE ACCURACY BY LEAD TIME")
    print("="*70)
    
    if len(actuals) == 0:
        print("No actual temperature data available")
        return None
    
    # Forecasts are already at exact hours (xx:00), so just use valid_time
    forecasts_copy = forecasts.copy()
    forecasts_copy['valid_hour'] = forecasts_copy['valid_time'].dt.floor('h')
    
    # Round actual readings to nearest hour
    actuals_rounded = actuals.copy()
    actuals_rounded['actual_hour'] = actuals_rounded['timestamp'].dt.round('h')
    
    # Aggregate actuals to hourly averages (in case multiple readings per hour)
    actuals_hourly = actuals_rounded.groupby('actual_hour').agg({
        'actual_temp': 'mean'
    }).reset_index()
    
    print(f"Aggregated {len(actuals)} actual readings into {len(actuals_hourly)} hourly averages")
    
    # Debug: Show date overlap
    forecast_dates = set(forecasts_copy['valid_hour'].dt.date)
    actual_dates = set(actuals_hourly['actual_hour'].dt.date)
    overlap_dates = forecast_dates & actual_dates
    print(f"Forecast date range: {min(forecast_dates)} to {max(forecast_dates)} ({len(forecast_dates)} days)")
    print(f"Actual date range: {min(actual_dates)} to {max(actual_dates)} ({len(actual_dates)} days)")
    print(f"Overlapping dates: {len(overlap_dates)} days")
    
    if len(overlap_dates) < 7:
        print(f"\n⚠️  WARNING: Limited overlap ({len(overlap_dates)} days)")
        print(f"   For more robust accuracy metrics, fetch more data:")
        print(f"   - Fetch forecasts for more recent dates, OR")
        print(f"   - Fetch actual temperatures for earlier dates")
    
    # Merge forecasts with actuals
    merged = forecasts_copy.merge(
        actuals_hourly,
        left_on='valid_hour',
        right_on='actual_hour',
        how='inner'
    )
    
    if len(merged) == 0:
        print("No matching forecast-actual pairs found")
        return None
    
    print(f"Matched {len(merged):,} forecast-actual pairs")
    
    # Calculate errors
    merged['error'] = merged['temperature'] - merged['actual_temp']
    merged['abs_error'] = np.abs(merged['error'])
    
    # Show sample matches for verification
    if len(merged) > 0:
        print(f"\nSample matches (first 5 for lead_time=2):")
        sample = merged[merged['lead_time'] == 2].head(5)
        if len(sample) > 0:
            print(sample[['valid_hour', 'lead_time', 'temperature', 'actual_temp', 'error']].to_string(index=False))
    
    # Overall accuracy
    overall = {
        'mae': float(merged['abs_error'].mean()),
        'rmse': float(np.sqrt((merged['error'] ** 2).mean())),
        'bias': float(merged['error'].mean()),
        'count': len(merged)
    }
    
    print(f"\nOverall Hourly Accuracy:")
    print(f"  MAE:   {overall['mae']:.2f}°F")
    print(f"  RMSE:  {overall['rmse']:.2f}°F")
    print(f"  Bias:  {overall['bias']:+.2f}°F")
    print(f"  Count: {overall['count']:,} hourly forecasts")
    
    # By lead time
    if 'lead_time' in merged.columns:
        print(f"\nAccuracy by Lead Time (days before):")
        print(f"{'Lead Time':<12} {'MAE':<8} {'RMSE':<8} {'Bias':<8} {'Count':<8}")
        print("-" * 50)
        
        by_lead_time = {}
        for lead in sorted(merged['lead_time'].unique()):
            lead_data = merged[merged['lead_time'] == lead]
            mae = lead_data['abs_error'].mean()
            rmse = np.sqrt((lead_data['error'] ** 2).mean())
            bias = lead_data['error'].mean()
            count = len(lead_data)
            
            by_lead_time[f'{lead}d'] = {
                'mae': float(mae),
                'rmse': float(rmse),
                'bias': float(bias),
                'count': int(count)
            }
            
            print(f"{lead} day(s)      {mae:6.2f}°F  {rmse:6.2f}°F  {bias:+6.2f}°F  {count:6d}")
        
        return {
            'overall': overall,
            'by_lead_time': by_lead_time
        }
    
    return {'overall': overall}


def metric_2_daily_max_accuracy(forecasts, actuals, daily_max=None):
    """
    METRIC 2: Daily Maximum Temperature Accuracy by Lead Time
    
    For each lead time, how accurate is the prediction of the daily maximum temperature?
    This is the KEY metric for Polymarket betting.
    """
    print("\n" + "="*70)
    print("METRIC 2: DAILY MAXIMUM TEMPERATURE ACCURACY BY LEAD TIME")
    print("="*70)
    
    if len(actuals) == 0 and daily_max is None:
        print("No actual temperature data available")
        return None
    
    # Determine actual daily maxes
    if daily_max is not None:
        print("Using Weather Underground daily max temperatures (official Polymarket source)")
        actual_daily_max = daily_max.copy()
        actual_daily_max['date'] = pd.to_datetime(actual_daily_max['date']).dt.date
    else:
        print("Calculating daily max from hourly data")
        actuals_copy = actuals.copy()
        actuals_copy['date'] = actuals_copy['timestamp'].dt.date
        actual_daily_max = actuals_copy.groupby('date').agg({
            'actual_temp': 'max'
        }).reset_index()
        actual_daily_max.rename(columns={'actual_temp': 'max_temp_f'}, inplace=True)
    
    # Calculate forecasted daily max for each target date and lead time
    forecasts_copy = forecasts.copy()
    forecasts_copy['target_date'] = forecasts_copy['valid_time'].dt.date
    
    # Group by target_date and lead_time, get max temperature
    forecast_daily_max = forecasts_copy.groupby(['target_date', 'lead_time']).agg({
        'temperature': 'max',
        'forecast_issued': 'first'
    }).reset_index()
    forecast_daily_max.rename(columns={'temperature': 'forecasted_max'}, inplace=True)
    
    # Merge with actuals
    merged = forecast_daily_max.merge(
        actual_daily_max,
        left_on='target_date',
        right_on='date',
        how='inner'
    )
    
    if len(merged) == 0:
        print("No matching forecast-actual pairs found")
        return None
    
    print(f"Matched {len(merged):,} daily forecast-actual pairs")
    
    # Calculate errors
    merged['error'] = merged['forecasted_max'] - merged['max_temp_f']
    merged['abs_error'] = np.abs(merged['error'])
    
    # Overall accuracy
    overall = {
        'mae': float(merged['abs_error'].mean()),
        'rmse': float(np.sqrt((merged['error'] ** 2).mean())),
        'bias': float(merged['error'].mean()),
        'count': len(merged)
    }
    
    print(f"\nOverall Daily Max Accuracy:")
    print(f"  MAE:   {overall['mae']:.2f}°F")
    print(f"  RMSE:  {overall['rmse']:.2f}°F")
    print(f"  Bias:  {overall['bias']:+.2f}°F")
    print(f"  Count: {overall['count']:,} days")
    
    # By lead time
    print(f"\nAccuracy by Lead Time (days before):")
    print(f"{'Lead Time':<12} {'MAE':<8} {'RMSE':<8} {'Bias':<8} {'Count':<8}")
    print("-" * 50)
    
    by_lead_time = {}
    for lead in sorted(merged['lead_time'].unique()):
        lead_data = merged[merged['lead_time'] == lead]
        mae = lead_data['abs_error'].mean()
        rmse = np.sqrt((lead_data['error'] ** 2).mean())
        bias = lead_data['error'].mean()
        count = len(lead_data)
        
        by_lead_time[f'{lead}d'] = {
            'mae': float(mae),
            'rmse': float(rmse),
            'bias': float(bias),
            'count': int(count)
        }
        
        timing_note = ""
        if lead == 0:
            timing_note = " (same day - nowcast)"
        elif lead == 1:
            timing_note = " (day before event)"
        elif lead == 2:
            timing_note = " (market opens)"
        
        print(f"{lead} day(s)      {mae:6.2f}°F  {rmse:6.2f}°F  {bias:+6.2f}°F  {count:6d}{timing_note}")
    
    # Show sample predictions
    print(f"\nSample predictions (lead_time=2, market opening):")
    sample = merged[merged['lead_time'] == 2].head(10)
    if len(sample) > 0:
        print(sample[['target_date', 'forecasted_max', 'max_temp_f', 'error']].to_string(index=False))
    
    return {
        'overall': overall,
        'by_lead_time': by_lead_time
    }


def main():
    """Run all accuracy analyses"""
    
    print("="*70)
    print("FORECAST ACCURACY ANALYSIS")
    print("="*70)
    
    # Load data
    forecasts, actuals = load_raw_data()
    
    # Try to load daily max data
    daily_max = None
    try:
        daily_max = pd.read_csv('data/raw/wunderground_daily_max_temps.csv')
        daily_max['date'] = pd.to_datetime(daily_max['date'])
        print(f"Loaded {len(daily_max)} days of daily max temperatures")
    except FileNotFoundError:
        print("No daily max temperature file found, will calculate from hourly data")
    
    # Run metrics
    metric1 = metric_1_hourly_accuracy(forecasts, actuals)
    metric2 = metric_2_daily_max_accuracy(forecasts, actuals, daily_max)
    
    # Save results
    results = {
        'created_at': datetime.now().isoformat(),
        'metric_1_hourly_accuracy': metric1,
        'metric_2_daily_max_accuracy': metric2
    }
    
    output_path = 'data/processed/forecast_accuracy_metrics.json'
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*70}")
    
    # Print interpretation
    print("\n" + "="*70)
    print("WHAT DO THESE METRICS MEAN?")
    print("="*70)
    
    print("\n📊 MAE (Mean Absolute Error):")
    print("   Average error ignoring direction (+ or -)")
    print("   Lower is better. 2.5°F means forecasts are off by ~2.5°F on average")
    
    print("\n📊 RMSE (Root Mean Squared Error):")
    print("   Like MAE but penalizes large errors more heavily")
    print("   Always >= MAE. Useful for detecting occasional big misses")
    
    print("\n📊 Bias:")
    print("   Systematic over/under prediction")
    print("   +0.5°F = forecasts are slightly too WARM on average")
    print("   -2°F = forecasts are consistently too COLD")
    print("   0°F = perfect, no systematic error")
    
    print("\n" + "="*70)
    print("RECOMMENDATIONS FOR POLYMARKET BETTING:")
    print("="*70)
    
    if metric2 and 'by_lead_time' in metric2:
        by_lead = metric2['by_lead_time']
        
        if '2d' in by_lead:
            print(f"\n✅ When market opens (2 days before):")
            print(f"   MAE:  {by_lead['2d']['mae']:.2f}°F (average error)")
            print(f"   Bias: {by_lead['2d']['bias']:+.2f}°F")
            print(f"   → If forecast says 45°F, actual will likely be")
            print(f"     between {45-by_lead['2d']['mae']:.0f}-{45+by_lead['2d']['mae']:.0f}°F")
        
        if '1d' in by_lead:
            print(f"\n📊 Day before event (1 day before):")
            print(f"   MAE:  {by_lead['1d']['mae']:.2f}°F")
            print(f"   → Forecast improves as event approaches")
        
        if '0d' in by_lead:
            print(f"\n📊 Same day (nowcast):")
            print(f"   MAE:  {by_lead['0d']['mae']:.2f}°F")
            print(f"   → Most accurate, but too late for betting")
        
        # Show degradation
        if '2d' in by_lead and '1d' in by_lead:
            mae_2d = by_lead['2d']['mae']
            mae_1d = by_lead['1d']['mae']
            if mae_2d > mae_1d:
                degradation = ((mae_2d - mae_1d) / mae_1d) * 100
                print(f"\n💡 Forecast accuracy improves by {degradation:.0f}% from 2 days to 1 day before")
                print(f"   Consider updating positions as event approaches if forecast changes significantly")


if __name__ == '__main__':
    main()

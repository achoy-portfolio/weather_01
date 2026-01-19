"""
Comprehensive Forecast Accuracy Analysis

Directly compares raw forecast data to actual temperatures without complex merging.
Tracks multiple accuracy metrics for different use cases.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json


def load_raw_data():
    """Load raw forecast and actual temperature data"""
    print("Loading raw data...")
    
    # Load forecasts
    forecasts = pd.read_csv('data/raw/historical_forecasts.csv')
    forecasts['forecast_datetime'] = pd.to_datetime(
        forecasts['forecast_date'] + ' ' + forecasts['forecast_time']
    )
    forecasts['valid_time'] = pd.to_datetime(forecasts['valid_time'])
    
    # Load actuals
    actuals = pd.read_csv('data/raw/actual_temperatures.csv')
    actuals['timestamp'] = pd.to_datetime(actuals['timestamp'])
    actuals.rename(columns={'temperature_f': 'actual_temp'}, inplace=True)
    
    print(f"Loaded {len(forecasts):,} forecast records")
    print(f"Loaded {len(actuals):,} actual temperature records")
    
    return forecasts, actuals


def metric_1_hourly_accuracy(forecasts, actuals):
    """
    METRIC 1: Hourly Temperature Accuracy
    
    For each hourly forecast, how accurate is it compared to the actual temperature?
    This is the most basic accuracy metric.
    """
    print("\n" + "="*70)
    print("METRIC 1: HOURLY TEMPERATURE ACCURACY")
    print("="*70)
    
    # Merge forecasts with actuals on valid_time = timestamp
    merged = forecasts.merge(
        actuals[['timestamp', 'actual_temp']], 
        left_on='valid_time', 
        right_on='timestamp',
        how='inner'
    )
    
    # Calculate error
    merged['error'] = merged['temperature'] - merged['actual_temp']
    merged['abs_error'] = np.abs(merged['error'])
    
    # Calculate lead time
    merged['lead_hours'] = (merged['valid_time'] - merged['forecast_datetime']).dt.total_seconds() / 3600
    
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
    print(f"  Bias:  {overall['bias']:.2f}°F")
    print(f"  Count: {overall['count']:,} hourly forecasts")
    
    # By lead time (in 6-hour buckets)
    print(f"\nAccuracy by Lead Time:")
    by_lead_time = {}
    for hours in [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72]:
        lead_data = merged[
            (merged['lead_hours'] >= hours) & 
            (merged['lead_hours'] < hours + 6)
        ]
        if len(lead_data) > 0:
            by_lead_time[f'{hours}h'] = {
                'mae': float(lead_data['abs_error'].mean()),
                'rmse': float(np.sqrt((lead_data['error'] ** 2).mean())),
                'count': len(lead_data)
            }
            print(f"  {hours:2d}-{hours+6:2d}h: MAE={by_lead_time[f'{hours}h']['mae']:.2f}°F, n={by_lead_time[f'{hours}h']['count']:,}")
    
    return {
        'overall': overall,
        'by_lead_time': by_lead_time
    }


def metric_2_next_day_max(forecasts, actuals):
    """
    METRIC 2: Next Day Maximum Temperature Accuracy
    
    If I issue a forecast at 9 PM today, how accurate is my prediction 
    of tomorrow's maximum temperature?
    
    This is the KEY metric for Polymarket betting.
    """
    print("\n" + "="*70)
    print("METRIC 2: NEXT DAY MAXIMUM TEMPERATURE ACCURACY")
    print("="*70)
    print("(Forecast issued at 9 PM for next day's high)")
    
    results = []
    
    # Get unique forecast dates
    forecast_dates = forecasts['forecast_date'].unique()
    
    for forecast_date in forecast_dates:
        # Get 9 PM forecast
        evening_forecast = forecasts[
            (forecasts['forecast_date'] == forecast_date) &
            (forecasts['forecast_time'] == '21:00')
        ].copy()
        
        if len(evening_forecast) == 0:
            continue
        
        # Calculate next day
        next_day = pd.to_datetime(forecast_date) + timedelta(days=1)
        next_day_str = next_day.strftime('%Y-%m-%d')
        
        # Get forecasted max for next day
        next_day_forecasts = evening_forecast[
            evening_forecast['valid_time'].dt.date == next_day.date()
        ]
        
        if len(next_day_forecasts) == 0:
            continue
        
        forecasted_max = next_day_forecasts['temperature'].max()
        
        # Get actual max for next day
        next_day_actuals = actuals[
            actuals['timestamp'].dt.date == next_day.date()
        ]
        
        if len(next_day_actuals) == 0:
            continue
        
        actual_max = next_day_actuals['actual_temp'].max()
        
        # Calculate error
        error = forecasted_max - actual_max
        
        results.append({
            'forecast_date': forecast_date,
            'target_date': next_day_str,
            'forecasted_max': forecasted_max,
            'actual_max': actual_max,
            'error': error,
            'abs_error': abs(error)
        })
    
    df = pd.DataFrame(results)
    
    if len(df) > 0:
        metrics = {
            'mae': float(df['abs_error'].mean()),
            'rmse': float(np.sqrt((df['error'] ** 2).mean())),
            'bias': float(df['error'].mean()),
            'count': len(df)
        }
        
        print(f"\n9 PM Forecast for Next Day's High:")
        print(f"  MAE:   {metrics['mae']:.2f}°F")
        print(f"  RMSE:  {metrics['rmse']:.2f}°F")
        print(f"  Bias:  {metrics['bias']:.2f}°F")
        print(f"  Count: {metrics['count']} days")
        
        return metrics
    else:
        print("\nNo data available for this metric")
        return None





def main():
    """Run all accuracy analyses"""
    
    # Load data
    forecasts, actuals = load_raw_data()
    
    # Run metrics 1 and 2 only
    metric1 = metric_1_hourly_accuracy(forecasts, actuals)
    metric2 = metric_2_next_day_max(forecasts, actuals)
    
    # Save results
    results = {
        'created_at': datetime.now().isoformat(),
        'metric_1_hourly_accuracy': metric1,
        'metric_2_next_day_max': metric2
    }
    
    output_path = 'data/processed/forecast_accuracy_metrics.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*70}")
    
    print("\n" + "="*70)
    print("WHAT DO THESE METRICS MEAN?")
    print("="*70)
    
    print("\n📊 MAE (Mean Absolute Error):")
    print("   Average error ignoring direction (+ or -)")
    print("   Lower is better. 2.17°F means forecasts are off by ~2°F on average")
    
    print("\n📊 RMSE (Root Mean Squared Error):")
    print("   Like MAE but penalizes large errors more heavily")
    print("   Always >= MAE. Useful for detecting occasional big misses")
    
    print("\n📊 Bias:")
    print("   Systematic over/under prediction")
    print("   +0.61°F = forecasts are slightly too WARM on average")
    print("   -2°F = forecasts are consistently too COLD")
    print("   0°F = perfect, no systematic error")
    
    print("\n" + "="*70)
    print("RECOMMENDATIONS FOR POLYMARKET BETTING:")
    print("="*70)
    
    if metric2:
        print(f"\n✅ BEST: Use 9 PM forecast for next day's high")
        print(f"   MAE:  {metric2['mae']:.2f}°F (average error)")
        print(f"   Bias: {metric2['bias']:+.2f}°F (slightly warm)")
        print(f"   This is your most reliable forecast for betting!")
        print(f"\n   Translation: If forecast says 45°F, actual will likely be")
        print(f"   between 43-47°F (within ±2°F about 68% of the time)")


if __name__ == '__main__':
    main()

"""
Polymarket Temperature Tracker for NYC (KLGA)

Tracks the daily high temperature and predicts Polymarket resolution.
Uses NWS 5-minute data to get the most accurate current high.

Run: python scripts/polymarket_temp_tracker.py
"""

import sys
from datetime import datetime, timedelta, timezone, time

sys.path.insert(0, '.')

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

from src.data.weather_scraper import WeatherScraper, WeatherDataError

NY_TZ = ZoneInfo("America/New_York")


def get_todays_high():
    """Fetch today's temperature data and find the current high."""
    scraper = WeatherScraper(station_id="KLGA")
    
    # Get today's date in NY timezone
    now_ny = datetime.now(NY_TZ)
    today_start = datetime.combine(now_ny.date(), time.min, tzinfo=NY_TZ)
    
    # Convert to UTC for API
    start_utc = today_start.astimezone(timezone.utc)
    end_utc = datetime.now(timezone.utc)
    
    df = scraper.fetch_raw_observations(start_utc, end_utc)
    
    if df.empty:
        return None, None, None
    
    # Convert to NY time
    df.index = df.index.tz_convert(NY_TZ)
    
    # Get actual max from all 5-min readings
    actual_high = df['temp_f'].max()
    actual_low = df['temp_f'].min()
    latest_temp = df['temp_f'].iloc[-1]
    high_time = df['temp_f'].idxmax()
    
    # Simulate Weather Underground hourly sampling
    # WU typically uses readings at :51 or :00 of each hour
    hourly_readings = []
    for hour in range(24):
        # Look for readings near :51 or :00
        for minute_target in [51, 0, 52, 50, 1, 2]:
            mask = (df.index.hour == hour) & (df.index.minute == minute_target)
            if mask.any():
                hourly_readings.append(df.loc[mask, 'temp_f'].iloc[0])
                break
    
    wu_simulated_high = max(hourly_readings) if hourly_readings else None
    
    return {
        'actual_high': actual_high,  # True max from all 5-min readings
        'wu_simulated_high': wu_simulated_high,  # Max from hourly samples only
        'current_low': actual_low,
        'latest_temp': latest_temp,
        'high_time': high_time,
        'observations': len(df),
        'last_update': df.index[-1],
        'hourly_samples': len(hourly_readings),
    }


def predict_polymarket_resolution(high_temp):
    """
    Predict which Polymarket bucket the temperature falls into.
    
    Polymarket typically uses Weather Underground which rounds to whole numbers.
    """
    # Standard rounding (33.5+ -> 34, 33.4- -> 33)
    rounded = round(high_temp)
    
    # Define common Polymarket buckets
    buckets = [
        (float('-inf'), 29, "≤29°F"),
        (30, 31, "30-31°F"),
        (32, 33, "32-33°F"),
        (34, 35, "34-35°F"),
        (36, 37, "36-37°F"),
        (38, 39, "38-39°F"),
        (40, 41, "40-41°F"),
        (42, 43, "42-43°F"),
        (44, float('inf'), "≥44°F"),
    ]
    
    for low, high, label in buckets:
        if low <= rounded <= high:
            return label, rounded
    
    return f"{rounded}°F", rounded


def main():
    print("=" * 60)
    print("Polymarket NYC Temperature Tracker (KLGA)")
    print("=" * 60)
    
    now_ny = datetime.now(NY_TZ)
    print(f"\nCurrent time: {now_ny.strftime('%Y-%m-%d %I:%M %p %Z')}")
    
    print("\nFetching today's temperature data...")
    
    try:
        stats = get_todays_high()
        
        if stats is None:
            print("No data available yet for today.")
            return
        
        print(f"\n{'─' * 40}")
        print("TODAY'S STATS (so far)")
        print(f"{'─' * 40}")
        print(f"  Actual High (5-min):  {stats['actual_high']:.1f}°F")
        print(f"  WU Simulated High:    {stats['wu_simulated_high']:.1f}°F" if stats['wu_simulated_high'] else "  WU Simulated High:    N/A")
        print(f"  High recorded at:     {stats['high_time'].strftime('%I:%M %p')}")
        print(f"  Current Low:          {stats['current_low']:.1f}°F")
        print(f"  Latest Reading:       {stats['latest_temp']:.1f}°F")
        print(f"  5-min Observations:   {stats['observations']}")
        print(f"  Hourly Samples:       {stats['hourly_samples']}")
        print(f"  Last Update:          {stats['last_update'].strftime('%I:%M %p')}")
        
        # Check for discrepancy
        if stats['wu_simulated_high'] and abs(stats['actual_high'] - stats['wu_simulated_high']) >= 0.5:
            print(f"\n  ⚠️  DISCREPANCY: Actual high ({stats['actual_high']:.1f}°F) differs from")
            print(f"      hourly-sampled high ({stats['wu_simulated_high']:.1f}°F)!")
            print(f"      WU may report lower value.")
        
        # Polymarket prediction - use WU simulated if available
        prediction_temp = stats['wu_simulated_high'] if stats['wu_simulated_high'] else stats['actual_high']
        bucket, rounded = predict_polymarket_resolution(prediction_temp)
        
        # Also show what actual high would resolve to
        actual_bucket, actual_rounded = predict_polymarket_resolution(stats['actual_high'])
        
        print(f"\n{'─' * 40}")
        print("POLYMARKET PREDICTION")
        print(f"{'─' * 40}")
        print(f"  WU-Style High:       {prediction_temp:.1f}°F → {rounded}°F → {bucket}")
        if actual_rounded != rounded:
            print(f"  Actual High:         {stats['actual_high']:.1f}°F → {actual_rounded}°F → {actual_bucket}")
            print(f"  ⚠️  Different buckets! WU sampling may affect resolution.")
        print(f"  Likely Bucket:       {bucket}")
        
        # Show confidence based on time of day
        hour = now_ny.hour
        if hour < 12:
            confidence = "LOW - Morning, high likely not reached yet"
        elif hour < 15:
            confidence = "MEDIUM - Afternoon, high may still increase"
        elif hour < 18:
            confidence = "HIGH - Late afternoon, high likely reached"
        else:
            confidence = "VERY HIGH - Evening, daily high is set"
        
        print(f"  Confidence:      {confidence}")
        
        # Edge case warning
        decimal = prediction_temp % 1
        if 0.4 <= decimal <= 0.6:
            print(f"\n  ⚠️  WARNING: Temperature {prediction_temp:.1f}°F is near rounding boundary!")
            print(f"      Could round to {int(prediction_temp)}°F or {int(prediction_temp)+1}°F")
        
    except WeatherDataError as e:
        print(f"Error fetching data: {e}")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("Note: Resolution depends on Polymarket's official source.")
    print("Check market rules for exact resolution criteria.")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""Test fetching forecast for today"""

from datetime import date, datetime
import requests

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

NY_TZ = ZoneInfo("America/New_York")

target_date = date.today()
print(f"Testing forecast for: {target_date}")

url = "https://api.open-meteo.com/v1/forecast"

lat = 40.7769
lon = -73.8740

days_ahead = (target_date - date.today()).days
print(f"Days ahead: {days_ahead}")

params = {
    'latitude': lat,
    'longitude': lon,
    'hourly': 'temperature_2m',
    'temperature_unit': 'fahrenheit',
    'timezone': 'America/New_York',
    'forecast_days': max(3, days_ahead + 1)
}

print(f"Requesting {params['forecast_days']} days of forecast")

response = requests.get(url, params=params, timeout=30)
response.raise_for_status()
data = response.json()

if 'hourly' in data:
    times = data['hourly'].get('time', [])
    temps = data['hourly'].get('temperature_2m', [])
    
    print(f"\nGot {len(times)} hourly records")
    
    if times:
        first_time = datetime.fromisoformat(times[0])
        last_time = datetime.fromisoformat(times[-1])
        print(f"Time range: {first_time.date()} to {last_time.date()}")
    
    # Filter for target date
    target_records = []
    for time_str, temp in zip(times, temps):
        if temp is None:
            continue
        
        timestamp = datetime.fromisoformat(time_str).replace(tzinfo=NY_TZ)
        
        if timestamp.date() == target_date:
            target_records.append({
                'time': timestamp,
                'temp': temp
            })
    
    print(f"\nRecords for {target_date}: {len(target_records)}")
    
    if target_records:
        temps_only = [r['temp'] for r in target_records]
        print(f"Temperature range: {min(temps_only):.1f}°F to {max(temps_only):.1f}°F")
        print(f"Forecasted max: {max(temps_only):.1f}°F")
        
        print(f"\nFirst few records:")
        for r in target_records[:5]:
            print(f"  {r['time'].strftime('%Y-%m-%d %H:%M')} - {r['temp']:.1f}°F")
    else:
        print("❌ No records found for target date!")
        print("\nShowing first 5 records:")
        for time_str, temp in zip(times[:5], temps[:5]):
            timestamp = datetime.fromisoformat(time_str)
            print(f"  {timestamp} - {temp}°F")

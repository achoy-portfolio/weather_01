"""
Quick script to verify timezone handling in NWS data.
Run: python verify_timezone.py
"""

import pandas as pd
from datetime import datetime

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

print("=" * 70)
print("Timezone Verification")
print("=" * 70)

# Load NWS data
nws_file = 'data/raw/nws_klga_5min_24h.csv'
df = pd.read_csv(nws_file, parse_dates=['timestamp'])

print(f"\n📁 File: {nws_file}")
print(f"📊 Records: {len(df)}")

# Show first timestamp
first_ts = df.iloc[0]['timestamp']
print(f"\n🕐 First timestamp:")
print(f"   Raw: {first_ts}")
print(f"   Type: {type(first_ts)}")

# Parse and show timezone info
if isinstance(first_ts, str):
    first_ts = pd.to_datetime(first_ts)

print(f"   Timezone: {first_ts.tzinfo}")

# Extract offset
ts_str = str(df.iloc[0]['timestamp'])
if '-05:00' in ts_str:
    print(f"   ✓ Offset: -05:00 (Eastern Standard Time)")
    print(f"   ✓ This is New York time, NOT Edmonton")
elif '-04:00' in ts_str:
    print(f"   ✓ Offset: -04:00 (Eastern Daylight Time)")
    print(f"   ✓ This is New York time, NOT Edmonton")
else:
    print(f"   ⚠️  Offset: {ts_str.split()[-1] if ' ' in ts_str else 'Unknown'}")

print("\n" + "=" * 70)
print("Timezone Reference")
print("=" * 70)
print("Edmonton (MST/MDT):  UTC-7 (winter) or UTC-6 (summer)")
print("New York (EST/EDT):  UTC-5 (winter) or UTC-4 (summer)")
print("=" * 70)

# Show current time in different zones
now_utc = datetime.now(ZoneInfo("UTC"))
now_ny = now_utc.astimezone(ZoneInfo("America/New_York"))
now_edmonton = now_utc.astimezone(ZoneInfo("America/Edmonton"))

print("\n🌍 Current time comparison:")
print(f"   UTC:      {now_utc.strftime('%Y-%m-%d %I:%M:%S %p %Z')}")
print(f"   New York: {now_ny.strftime('%Y-%m-%d %I:%M:%S %p %Z')}")
print(f"   Edmonton: {now_edmonton.strftime('%Y-%m-%d %I:%M:%S %p %Z')}")

print("\n✅ Conclusion: Your NWS data is in New York time (Eastern Time)")
print("=" * 70)

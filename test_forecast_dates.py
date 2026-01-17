import requests
from datetime import datetime, timedelta

print("Testing forecast dates...")
print(f"Today: {datetime.now().strftime('%Y-%m-%d')}")
print(f"Tomorrow: {(datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')}")
print()

# Test wttr.in
print("wttr.in:")
r = requests.get('https://wttr.in/40.7769,-73.874', params={'format': 'j1'})
d = r.json()
for i, w in enumerate(d['weather'][:3]):
    print(f"  Index {i}: {w['date']} - High: {w['maxtempF']}°F, Low: {w['mintempF']}°F")

print()

# Test 7Timer
print("7Timer:")
r = requests.get('https://www.7timer.info/bin/api.pl', params={
    'lon': -73.874, 'lat': 40.7769, 'product': 'civil', 'output': 'json'
})
d = r.json()
print(f"  Init time: {d['init']}")
for i, period in enumerate(d['dataseries'][:12]):
    temp_c = period['temp2m']
    temp_f = temp_c * 9/5 + 32
    print(f"  Period {i} (timepoint {period['timepoint']}h): {temp_f:.1f}°F")

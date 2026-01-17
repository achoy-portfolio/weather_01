# Temperature vs Odds Analysis

## Quick Answer to Your Questions

### 1. **Timezone Verification** ✅

Your NWS data shows `-05:00` which is **New York time (Eastern Standard Time)**, NOT Edmonton time.

- Edmonton is Mountain Time (MST/MDT): UTC-7 or UTC-6
- New York is Eastern Time (EST/EDT): UTC-5 or UTC-4
- The timestamps are correct!

### 2. **Polymarket Historical Odds** ✅

Yes! Polymarket provides historical odds data through their CLOB API. I've created tools to fetch and analyze it.

## Understanding the Discrepancy

If you saw:

- **NWS reading**: 39.9°F at 9:51 AM
- **Odds increased**: at 8:00 AM

This could mean:

1. **Market anticipation**: Traders bet on warming trend before it happened
2. **Forecast data**: Traders saw weather forecasts predicting higher temps
3. **Time lag**: Odds reflect future expectations, not current readings
4. **Different data**: Traders may use multiple sources (not just NWS)

## New Tools Created

### 1. Historical Odds Fetcher

```bash
python scripts/fetching/fetch_polymarket_historical.py
```

**What it does:**

- Fetches historical odds for all temperature thresholds
- Downloads tick-by-tick price changes (5-minute intervals)
- Saves to `data/raw/polymarket_odds_history.csv`

**Options:**

- `interval`: '1m', '1h', '6h', '1d', '1w', 'max'
- `fidelity`: Resolution in minutes (5 = 5-minute data)

### 2. Combined Dashboard

```bash
streamlit run odds_vs_temperature_dashboard.py
```

**Features:**

- Side-by-side comparison of temperature and odds
- See exactly when odds changed vs when temp changed
- Timezone verification display
- Identify biggest odds movements
- Match temperature readings to odds changes

### 3. NWS Explorer Dashboard

```bash
streamlit run nws_explorer_dashboard.py
```

**Features:**

- Google Finance-style interactive chart
- Zoom and pan through temperature data
- Multiple time ranges (6h, 12h, 24h, 1w, 2w)
- Candlestick view for temperature ranges
- Additional metrics (humidity, wind, pressure)

## Workflow for Analysis

### Step 1: Fetch Historical Odds

```bash
# Get odds history for tomorrow's market
python scripts/fetching/fetch_polymarket_historical.py
```

This creates: `data/raw/polymarket_odds_history.csv`

### Step 2: Launch Combined Dashboard

```bash
streamlit run odds_vs_temperature_dashboard.py
```

### Step 3: Analyze

- Select your threshold (e.g., 40°F)
- Look at the dual chart:
  - Top: Temperature readings over time
  - Bottom: Odds changes over time
- Find when odds spiked
- Check what temperature was at that time
- Verify timezone in the "Timezone Verification" section

## Example Analysis

**Scenario**: Odds for ≥40°F increased at 8:00 AM, but temp was only 39.9°F at 9:51 AM

**Possible explanations:**

1. **Forward-looking**: At 8 AM, forecast showed warming trend
2. **Peak timing**: Market is for daily high (usually 2-4 PM), not current temp
3. **Information edge**: Some traders had better forecast data
4. **Volume spike**: Large bet moved the odds

**How to verify:**

1. Check NWS forecast at 8:00 AM (was it predicting 40°F+?)
2. Look at temperature trend from 8 AM to 2 PM
3. Check if odds continued rising or fell back
4. Compare to actual daily high

## Data Files

### Temperature Data

- `data/raw/nws_klga_5min_24h.csv` - Recent 5-minute readings
- `data/raw/klga_hourly_full_history.csv` - Historical hourly data

### Odds Data

- `data/raw/polymarket_odds_history.csv` - Historical odds (after running fetcher)
- `data/raw/polymarket_odds.json` - Current odds snapshot

## API Endpoints Used

### NWS API

- Endpoint: `https://api.weather.gov/stations/KLGA/observations`
- Returns: 5-minute interval observations
- Timezone: UTC (converted to ET in our code)

### Polymarket CLOB API

- Endpoint: `https://clob.polymarket.com/prices-history`
- Returns: Historical price data with configurable resolution
- Timezone: Unix timestamps (converted to ET in our code)

## Tips

1. **Always fetch fresh odds data** before analyzing - markets update constantly
2. **Match time ranges** - if analyzing 8 AM odds, look at temps from 8 AM onward
3. **Remember the market** - betting on "daily high", not current temperature
4. **Check volume** - low volume = odds can swing wildly on small trades
5. **Timezone matters** - all our tools use ET, but verify in the dashboard

## Troubleshooting

**No odds data showing?**

```bash
# Run the fetcher first
python scripts/fetching/fetch_polymarket_historical.py
```

**Wrong date?**

- The fetcher automatically gets tomorrow's market
- Edit the script to specify a different date

**Timezone confusion?**

- Check the "Timezone Verification" section in the dashboard
- All timestamps should show `-05:00` (EST) or `-04:00` (EDT)
- NWS API returns UTC, we convert to ET

**Odds seem wrong?**

- Polymarket shows probability as decimal (0.75 = 75%)
- YES token price = probability of outcome
- Check the question text to confirm threshold direction (≥ vs ≤)

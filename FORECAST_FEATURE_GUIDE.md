# Forecast Analysis Feature Guide

## What This Does

Shows what the weather forecast was at specific times when traders were placing bets. This helps you understand:

- What information traders had when odds changed
- How accurate forecasts were vs actual temperatures
- Whether traders were reacting to forecast updates

## Setup

### 1. Add Your Visual Crossing API Key

Update `.env`:

```
NOAA_API_KEY=WSPJBegSQBiSbKgtOTOsIqfZMfCLPaPx
VISUAL_CROSSING_API_KEY=your_actual_key_here
```

Get a free key at: https://www.visualcrossing.com/sign-up

**Free tier:** 1000 API calls/day (plenty for this use case)

### 2. Update Streamlit Secrets (for deployment)

In `.streamlit/secrets.toml`:

```toml
NOAA_API_KEY = "your_key"
VISUAL_CROSSING_API_KEY = "your_key"
```

## How to Use

### Step 1: Select a Market Date

Choose the date you want to analyze (e.g., Jan 17)

### Step 2: Enable Forecast Analysis

In the sidebar, check "Show historical forecast"

### Step 3: Select Forecast Time

Choose when the forecast was issued from the dropdown. This shows times when odds data exists, so you can see what forecast traders were looking at.

### Step 4: Analyze

The chart will show:

- **Gray dotted line**: Temperature before/after target date
- **Red solid line**: Actual temperature on target date
- **Purple dashed line**: What the forecast predicted
- **Purple vertical line**: When the forecast was issued

## Example Use Cases

### Use Case 1: Did traders react to forecast updates?

1. Select Jan 17 market
2. Enable forecast
3. Select 8:00 AM forecast time
4. Check if odds changed after this time
5. Compare forecast to actual temps

### Use Case 2: Was the forecast accurate?

1. Select a resolved market
2. Enable forecast
3. Select forecast from 2 days before
4. Compare purple line (forecast) to red line (actual)
5. See if traders should have trusted the forecast

### Use Case 3: Find arbitrage opportunities

1. Select current/future market
2. Enable forecast
3. Select latest forecast time
4. Compare forecast to current odds
5. If forecast shows 42°F but odds favor 40-41°F, there may be an edge

## Chart Legend

| Color  | Line Style    | Meaning                                |
| ------ | ------------- | -------------------------------------- |
| Gray   | Dotted        | Temperature (before/after target date) |
| Red    | Solid         | Actual temperature (target date)       |
| Purple | Dashed        | Forecast prediction                    |
| Purple | Vertical dot  | When forecast was issued               |
| Black  | Vertical dash | Target date boundaries                 |

## API Limits

**Visual Crossing Free Tier:**

- 1000 calls/day
- Each forecast request = 1 call
- Cached for 1 hour to save calls

**Tips to stay under limit:**

- Dashboard caches forecasts for 1 hour
- Only enable forecast when you need it
- Don't refresh unnecessarily

## Troubleshooting

**"Visual Crossing API key not configured"**

- Add your key to `.env` file
- Restart the dashboard

**"Could not load forecast data"**

- Check your API key is valid
- Verify you haven't hit the daily limit
- Check the date isn't too far in the past (Visual Crossing has limits)

**Forecast line doesn't show**

- Make sure "Show historical forecast" is checked
- Select a forecast time from the dropdown
- Wait for the data to load

## Advanced: Understanding the Data

The forecast shows:

- **3-day forward prediction** from the selected time
- **Hourly resolution** (24 data points per day)
- **Temperature only** (can be extended to show wind, precipitation, etc.)

The forecast is what traders would have seen at that specific time, helping you understand their decision-making process.

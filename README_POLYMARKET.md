# Polymarket Temperature Betting System

Automated system for finding +EV betting opportunities on Polymarket temperature markets.

## Overview

This system:

1. Trains probabilistic models on 20 years of KLGA weather data
2. Fetches current NWS forecasts
3. Scrapes Polymarket odds for temperature markets
4. Identifies positive expected value (EV) betting opportunities
5. Recommends bet sizes using Kelly Criterion

## Setup

### 1. Download Historical Data

```bash
python scripts/download_klga_full_history.py
```

This downloads 20 years of hourly weather data from Iowa Mesonet.

### 2. Train Probabilistic Model

```bash
python scripts/train_probabilistic_model.py
```

Trains quantile regression models (Q10, Q25, Q50, Q75, Q90) to estimate temperature distributions.

**Expected Performance:**

- 50% prediction interval coverage: ~50%
- 80% prediction interval coverage: ~80%
- Test set evaluation on 2023+ data

## Daily Workflow

### Option 1: Run Complete Pipeline

```bash
python scripts/polymarket_pipeline.py
```

This runs all steps in sequence:

1. Loads trained model
2. Prepares features from recent data
3. Fetches Polymarket odds
4. Analyzes opportunities
5. Recommends bets

### Option 2: Run Steps Manually

```bash
# 1. Fetch NWS forecast
python scripts/fetch_nws_forecast.py

# 2. Fetch Polymarket odds
python scripts/fetch_polymarket_odds.py

# 3. Run betting analysis
python scripts/betting_strategy.py
```

## Understanding the Output

### Prediction Summary

```
Predicted peak temperature: 45.2°F ± 3.5°F
80% confidence interval: [40.7°F, 49.7°F]
```

### Betting Opportunities

```
Threshold    Market     Model      Edge       EV         Kelly       Rec
>40°F        85.0%      92.0%      +7.0%      +8.2%      $45.50      *** BET
>45°F        60.0%      68.0%      +8.0%      +13.3%     $78.20      *** BET
>50°F        25.0%      18.0%      -7.0%      -28.0%     $0.00       PASS
```

**Columns:**

- **Threshold**: Temperature threshold for the market
- **Market**: Polymarket's implied probability
- **Model**: Your model's estimated probability
- **Edge**: Difference (Model - Market)
- **EV**: Expected value percentage
- **Kelly**: Recommended bet size (25% fractional Kelly)
- **Rec**: BET (if EV > 5%) or PASS

## Betting Strategy

### Kelly Criterion

- Uses **fractional Kelly (25%)** for risk management
- Only recommends bets with **EV > 5%** to account for model uncertainty
- Sizes bets proportionally to edge and bankroll

### Risk Management

1. **Never bet more than Kelly recommends**
2. **Track results** to calibrate model over time
3. **Start small** until you validate model accuracy
4. **Consider liquidity** - large bets may move the market

## Files Structure

```
scripts/
├── download_klga_full_history.py    # Download historical data
├── train_probabilistic_model.py     # Train quantile models
├── fetch_nws_forecast.py            # Get NWS forecast
├── fetch_polymarket_odds.py         # Scrape Polymarket
├── betting_strategy.py              # Betting analysis functions
└── polymarket_pipeline.py           # Complete pipeline

data/
├── raw/
│   ├── klga_hourly_full_history.csv # Historical weather data
│   ├── nws_forecast_klga.csv        # Current NWS forecast
│   └── polymarket_odds.csv          # Current market odds
└── results/
    └── betting_opportunities.csv    # Analysis results

models/
└── probabilistic_temp_model.pkl     # Trained quantile models
```

## Improving Accuracy

### Current Limitations

- ~5°F prediction error (MAE)
- Only uses historical patterns, not actual forecasts
- Simple quantile regression approach

### Potential Improvements

1. **Incorporate NWS forecast data** into model features
2. **Ensemble multiple models** (XGBoost + LightGBM + Neural Net)
3. **Add atmospheric features** (pressure, jet stream, etc.)
4. **Train separate models** for different seasons
5. **Use more sophisticated distribution estimation** (e.g., mixture models)

## Monitoring Performance

Track your bets in a spreadsheet:

- Date
- Market
- Threshold
- Your probability
- Market odds
- Bet size
- Outcome
- Profit/Loss

Calculate:

- **Win rate**: Should match your average predicted probability
- **ROI**: Should be positive if model has edge
- **Calibration**: Are your 70% predictions correct 70% of the time?

## Disclaimer

This is for educational purposes. Betting involves risk. Past performance doesn't guarantee future results. Always bet responsibly and within your means.

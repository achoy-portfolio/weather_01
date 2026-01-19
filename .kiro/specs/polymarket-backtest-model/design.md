# Design Document

## Overview

The Polymarket Backtest Model is a comprehensive system for evaluating historical betting performance using Open-Meteo forecast data and Polymarket odds. The system collects historical forecasts, actual temperature observations, and market odds to simulate betting decisions and build error models that quantify forecast uncertainty. This enables data-driven optimization of betting strategies by understanding when forecasts are reliable enough to justify placing bets.

The system operates in three phases:

1. **Data Collection**: Fetch historical forecasts, actual temperatures, and market odds
2. **Backtest Simulation**: Simulate betting decisions using historical data
3. **Error Modeling**: Build statistical models of forecast accuracy to inform future betting

## Architecture

### High-Level Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Backtest Pipeline                         │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Data       │  │   Backtest   │  │    Error     │     │
│  │  Collection  │─▶│  Simulation  │─▶│   Modeling   │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│         │                  │                  │             │
│         ▼                  ▼                  ▼             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  Raw Data    │  │   Results    │  │  Error Model │     │
│  │   (CSV)      │  │    (CSV)     │  │   (JSON)     │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

```
Open-Meteo Historical ──┐
Forecast API            │
                        ├──▶ Data Collection ──▶ Backtest ──▶ Error Model ──▶ Strategy
Open-Meteo Archive ─────┤         Module          Simulator      Builder       Report
API                     │
                        │
Polymarket API ─────────┘
```

## Components and Interfaces

### 1. Data Collection Module

**Purpose**: Fetch historical forecast data, actual temperatures, and market odds from external APIs.

**Sub-Components**:

#### 1.1 Historical Forecast Fetcher

**Responsibility**: Retrieve archived forecasts from Open-Meteo Historical Forecast API

**Key Functions**:

- `fetch_historical_forecast(target_date, forecast_time='21:00')`: Get forecast issued at specific time for target date
- `fetch_forecast_range(start_date, end_date)`: Batch fetch forecasts for date range
- `parse_forecast_response(response)`: Extract peak temperature from API response

**API Integration**:

```python
# Open-Meteo Historical Forecast API
url = "https://api.open-meteo.com/v1/historical-forecast-api"
params = {
    'latitude': 40.7769,
    'longitude': -73.8740,
    'start_date': target_date,
    'end_date': target_date,
    'daily': 'temperature_2m_max',
    'temperature_unit': 'fahrenheit',
    'timezone': 'America/New_York',
    'forecast_days': 1  # 1-day ahead forecast
}
```

**Key Design Decisions**:

- Focus on 9 PM (21:00) forecasts as they represent the last forecast before market close
- Use 1-day lead time (forecast issued day before target)
- Handle timezone conversion to ensure 9 PM Eastern Time
- Implement retry logic for API failures
- Cache responses to avoid redundant API calls

#### 1.2 Actual Temperature Fetcher

**Responsibility**: Retrieve observed temperature data from Open-Meteo Archive API

**Key Functions**:

- `fetch_actual_temperature(date)`: Get actual high, low, and average temps for date
- `fetch_actual_range(start_date, end_date)`: Batch fetch actuals for date range
- `calculate_daily_average(hourly_data)`: Compute average from hourly observations

**API Integration**:

```python
# Open-Meteo Archive API
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    'latitude': 40.7769,
    'longitude': -73.8740,
    'start_date': date,
    'end_date': date,
    'daily': 'temperature_2m_max,temperature_2m_min',
    'hourly': 'temperature_2m',
    'temperature_unit': 'fahrenheit',
    'timezone': 'America/New_York'
}
```

**Key Design Decisions**:

- Fetch both daily aggregates (max/min) and hourly data for average calculation
- Store all three metrics (high, low, average) for comprehensive analysis
- Handle missing data gracefully (some dates may not have complete observations)

#### 1.3 Polymarket Odds Fetcher

**Responsibility**: Retrieve historical market odds from Polymarket API

**Key Functions**:

- `fetch_market_odds(event_date, fetch_time=None)`: Get odds for all thresholds
- `fetch_odds_at_market_open(event_date)`: Get odds ~2 days before event
- `parse_threshold_buckets(markets)`: Extract temperature thresholds from market questions
- `convert_price_to_probability(price)`: Convert Polymarket prices to probabilities

**API Integration**:

```python
# Polymarket Event API
event_url = f"https://gamma-api.polymarket.com/events?slug={event_slug}"

# Polymarket Historical Prices API
price_url = "https://clob.polymarket.com/prices-history"
params = {
    'market': token_id,
    'startTs': market_open_timestamp,
    'endTs': event_timestamp,
    'fidelity': 60  # 1-hour resolution
}
```

**Key Design Decisions**:

- Use existing `fetch_polymarket_historical.py` as foundation
- Focus on odds at market open time (typically 2 days before event)
- Store odds for all threshold buckets (e.g., "≥75°F", "34-35°F", "≤33°F")
- Handle different threshold formats (ranges, "or higher", "or lower")
- Account for market resolution rules (rounding to whole degrees)

### 2. Backtest Simulation Module

**Purpose**: Simulate betting decisions using historical data and evaluate performance.

**Sub-Components**:

#### 2.1 Betting Decision Simulator

**Responsibility**: Determine which bets would have been placed based on forecasts and odds

**Key Functions**:

- `simulate_bet_decision(forecast, market_odds, threshold)`: Decide whether to bet
- `calculate_model_probability(forecast, threshold, uncertainty)`: Estimate win probability
- `calculate_expected_value(model_prob, market_odds)`: Compute EV
- `apply_kelly_criterion(ev, model_prob, market_odds, bankroll)`: Size bet

**Algorithm**:

```python
def simulate_bet_decision(forecast_temp, threshold, market_odds, uncertainty):
    # Calculate model probability using normal distribution
    z_score = (threshold - forecast_temp) / uncertainty
    model_prob = 1 - norm.cdf(z_score)  # P(temp >= threshold)

    # Calculate expected value
    payout_multiplier = 1 / market_odds
    ev = (model_prob * payout_multiplier) - 1

    # Only bet if EV > 5%
    if ev > 0.05:
        # Calculate Kelly bet size
        b = payout_multiplier - 1
        kelly = (b * model_prob - (1 - model_prob)) / b
        bet_size = kelly * 0.25 * bankroll  # Fractional Kelly

        return {
            'should_bet': True,
            'bet_size': bet_size,
            'expected_value': ev,
            'model_probability': model_prob
        }

    return {'should_bet': False}
```

**Key Design Decisions**:

- Use normal distribution assumption for temperature (reasonable for daily temps)
- Require minimum 5% EV to account for model uncertainty
- Apply fractional Kelly (25%) for risk management
- Simulate with fixed bankroll ($1000) for consistency

#### 2.2 Outcome Evaluator

**Responsibility**: Determine whether simulated bets would have won or lost

**Key Functions**:

- `evaluate_bet_outcome(bet, actual_temp, threshold)`: Check if bet won
- `calculate_profit_loss(bet, outcome)`: Compute P&L for bet
- `aggregate_results(all_bets)`: Summarize performance metrics

**Outcome Logic**:

```python
def evaluate_bet_outcome(bet_threshold, actual_temp, threshold_type):
    if threshold_type == 'above':  # e.g., "≥75°F"
        return actual_temp >= bet_threshold
    elif threshold_type == 'below':  # e.g., "≤33°F"
        return actual_temp <= bet_threshold
    elif threshold_type == 'range':  # e.g., "34-35°F"
        low, high = parse_range(bet_threshold)
        # Account for rounding: 34.5-35.4 rounds to 35°F
        return low - 0.5 <= actual_temp < high + 0.5
```

**Key Design Decisions**:

- Account for Polymarket rounding rules (whole degree resolution)
- Handle different threshold types (above, below, range)
- Calculate both win rate and ROI metrics
- Track performance by threshold type and lead time

### 3. Error Modeling Module

**Purpose**: Build statistical models of forecast accuracy to quantify uncertainty.

**Sub-Components**:

#### 3.1 Error Calculator

**Responsibility**: Compute forecast errors and statistical metrics

**Key Functions**:

- `calculate_forecast_error(forecast, actual)`: Compute error for single forecast
- `calculate_mae(errors)`: Mean Absolute Error
- `calculate_rmse(errors)`: Root Mean Squared Error
- `calculate_bias(errors)`: Mean Error (systematic bias)
- `calculate_error_distribution(errors)`: Fit distribution to errors

**Error Metrics**:

```python
# Basic error
error = forecast_temp - actual_temp

# Mean Absolute Error (MAE)
mae = mean(abs(errors))

# Root Mean Squared Error (RMSE)
rmse = sqrt(mean(errors^2))

# Bias (systematic over/under prediction)
bias = mean(errors)

# Standard deviation (uncertainty)
std_dev = std(errors)
```

#### 3.2 Uncertainty Estimator

**Responsibility**: Estimate forecast uncertainty for probability calculations

**Key Functions**:

- `estimate_uncertainty(lead_time, season, recent_errors)`: Predict uncertainty
- `fit_uncertainty_model(historical_errors)`: Build uncertainty model
- `get_confidence_interval(forecast, uncertainty, confidence_level)`: Calculate CI

**Uncertainty Model**:

```python
# Uncertainty varies by:
# 1. Lead time (longer = more uncertain)
# 2. Season (winter more variable than summer)
# 3. Recent forecast performance

def estimate_uncertainty(lead_time_days, month, recent_mae):
    # Base uncertainty from historical data
    base_uncertainty = historical_std_dev

    # Adjust for lead time
    lead_time_factor = 1 + (0.2 * lead_time_days)

    # Adjust for season (winter more variable)
    season_factor = 1.2 if month in [12, 1, 2] else 1.0

    # Adjust for recent performance
    performance_factor = recent_mae / historical_mae

    uncertainty = base_uncertainty * lead_time_factor * season_factor * performance_factor

    return uncertainty
```

**Key Design Decisions**:

- Use rolling window (last 30 days) for recent performance
- Separate error models by season and lead time
- Store error distributions for probabilistic forecasting
- Update uncertainty estimates as more data becomes available

#### 3.3 Strategy Optimizer

**Responsibility**: Use error model to optimize betting strategy parameters

**Key Functions**:

- `optimize_ev_threshold(error_model, historical_bets)`: Find optimal EV cutoff
- `optimize_kelly_fraction(error_model, historical_bets)`: Find optimal Kelly fraction
- `backtest_strategy_variants(error_model, odds_data)`: Test different strategies

**Optimization Approach**:

```python
# Test different strategy parameters
ev_thresholds = [0.03, 0.05, 0.07, 0.10]
kelly_fractions = [0.10, 0.25, 0.50]

best_roi = -float('inf')
best_params = None

for ev_thresh in ev_thresholds:
    for kelly_frac in kelly_fractions:
        # Simulate with these parameters
        results = backtest_with_params(ev_thresh, kelly_frac)

        if results['roi'] > best_roi and results['sharpe'] > 1.0:
            best_roi = results['roi']
            best_params = {'ev_threshold': ev_thresh, 'kelly_fraction': kelly_frac}

return best_params
```

## Data Models

### Forecast Record

```python
{
    'forecast_date': '2025-01-20',      # When forecast was issued
    'forecast_time': '21:00',           # Time forecast was issued (9 PM)
    'target_date': '2025-01-21',        # Date being forecasted
    'lead_time_days': 1,                # Days ahead
    'forecasted_high': 45.2,            # Predicted peak temp (°F)
    'forecasted_low': 32.1,             # Predicted low temp (°F)
    'source': 'open_meteo',             # Forecast source
    'model': 'GFS',                     # Weather model used
    'latitude': 40.7769,
    'longitude': -73.8740
}
```

### Actual Temperature Record

```python
{
    'date': '2025-01-21',
    'actual_high': 44.8,                # Observed peak temp (°F)
    'actual_low': 31.5,                 # Observed low temp (°F)
    'actual_average': 38.2,             # Average temp (°F)
    'peak_time': '14:35',               # When peak occurred
    'source': 'open_meteo_archive',
    'latitude': 40.7769,
    'longitude': -73.8740
}
```

### Market Odds Record

```python
{
    'event_date': '2025-01-21',
    'fetch_timestamp': '2025-01-19T21:00:00-05:00',  # When odds were captured
    'market_open_time': '2025-01-19T09:00:00-05:00', # When market opened
    'thresholds': [
        {
            'threshold': '≥75',
            'threshold_display': '≥75°F',
            'threshold_type': 'above',
            'yes_probability': 0.15,
            'no_probability': 0.85,
            'volume': 12500.00,
            'liquidity': 5000.00
        },
        {
            'threshold': '34-35',
            'threshold_display': '34-35°F',
            'threshold_type': 'range',
            'yes_probability': 0.25,
            'no_probability': 0.75,
            'volume': 8200.00,
            'liquidity': 3500.00
        }
    ]
}
```

### Backtest Result Record

```python
{
    'target_date': '2025-01-21',
    'forecast_temp': 45.2,
    'actual_temp': 44.8,
    'forecast_error': 0.4,              # forecast - actual
    'threshold': '≥45',
    'threshold_type': 'above',
    'market_odds': 0.55,
    'model_probability': 0.65,
    'edge': 0.10,                       # model_prob - market_odds
    'expected_value': 0.182,            # 18.2% EV
    'bet_placed': True,
    'bet_size': 58.50,
    'bet_outcome': 'loss',              # actual was 44.8, below 45
    'profit_loss': -58.50,
    'cumulative_pl': 125.30
}
```

### Error Model

```python
{
    'model_version': '1.0',
    'training_period': {
        'start_date': '2025-01-21',
        'end_date': '2025-02-15',
        'num_forecasts': 26
    },
    'overall_metrics': {
        'mae': 2.8,                     # Mean Absolute Error
        'rmse': 3.5,                    # Root Mean Squared Error
        'bias': -0.3,                   # Systematic bias (slight cold bias)
        'std_dev': 3.4                  # Standard deviation of errors
    },
    'by_lead_time': {
        '1_day': {'mae': 2.8, 'rmse': 3.5, 'std_dev': 3.4},
        '2_day': {'mae': 3.5, 'rmse': 4.2, 'std_dev': 4.0}
    },
    'by_season': {
        'winter': {'mae': 3.2, 'rmse': 4.0, 'std_dev': 3.8},
        'spring': {'mae': 2.5, 'rmse': 3.1, 'std_dev': 3.0}
    },
    'uncertainty_function': {
        'base_uncertainty': 3.4,
        'lead_time_coefficient': 0.2,
        'season_factors': {'winter': 1.2, 'spring': 1.0, 'summer': 0.9, 'fall': 1.1}
    }
}
```

## Error Handling

### API Failures

- **Retry Logic**: Exponential backoff with max 3 retries
- **Timeout Handling**: 10-second timeout per request
- **Rate Limiting**: Respect API rate limits (Open-Meteo is unlimited, but add delays)
- **Fallback**: Continue processing other dates if one fails

### Missing Data

- **Forecast Missing**: Log warning, skip that date in backtest
- **Actual Missing**: Log warning, cannot evaluate that date
- **Odds Missing**: Log warning, skip that market
- **Partial Data**: Use what's available, mark record as incomplete

### Data Quality Issues

- **Outlier Detection**: Flag forecasts/actuals that differ by >20°F from recent average
- **Validation**: Check that temperatures are within reasonable bounds (-20°F to 120°F for NYC)
- **Consistency**: Verify that high >= low, and average is between them
- **Rounding**: Apply Polymarket rounding rules consistently

### Error Logging

```python
# Log structure
{
    'timestamp': '2025-01-20T15:30:00',
    'component': 'HistoricalForecastFetcher',
    'error_type': 'APITimeout',
    'message': 'Timeout fetching forecast for 2025-01-21',
    'details': {'url': '...', 'params': {...}},
    'retry_count': 2,
    'resolved': False
}
```

## Testing Strategy

### Unit Tests

- **Data Fetchers**: Mock API responses, test parsing logic
- **Error Calculations**: Test MAE, RMSE, bias calculations with known data
- **Probability Calculations**: Verify normal distribution calculations
- **Kelly Criterion**: Test bet sizing with various inputs
- **Outcome Evaluation**: Test win/loss logic for different threshold types

### Integration Tests

- **End-to-End Pipeline**: Run backtest on small date range (5 days)
- **API Integration**: Test actual API calls (with rate limiting)
- **Data Persistence**: Verify CSV files are created correctly
- **Error Model Building**: Test with synthetic data

### Validation Tests

- **Forecast Accuracy**: Compare error metrics to expected ranges (MAE should be 2-5°F)
- **Probability Calibration**: Check that 70% predictions win ~70% of time
- **ROI Reasonableness**: Verify backtest ROI is within plausible range (-50% to +100%)
- **Data Completeness**: Ensure no missing dates in output

### Performance Tests

- **Batch Processing**: Time fetching 30 days of data
- **Memory Usage**: Monitor memory with large datasets
- **API Rate Limits**: Ensure we don't exceed limits

## File Structure

```
scripts/
├── fetching/
│   ├── fetch_historical_forecasts.py      # New: Fetch Open-Meteo historical forecasts
│   ├── fetch_actual_temperatures.py       # New: Fetch actual temps from archive
│   └── fetch_polymarket_historical.py     # Existing: Fetch historical odds
│
├── analysis/
│   ├── backtest_simulator.py              # New: Simulate betting decisions
│   ├── error_model_builder.py             # New: Build forecast error models
│   └── strategy_optimizer.py              # New: Optimize betting parameters
│
└── pipelines/
    └── run_backtest_pipeline.py            # New: Main pipeline orchestrator

data/
├── raw/
│   ├── historical_forecasts.csv            # Forecasts from Open-Meteo
│   ├── actual_temperatures.csv             # Actual temps from archive
│   └── polymarket_odds_history.csv         # Market odds (existing)
│
├── processed/
│   ├── backtest_data_combined.csv          # Merged forecast + actual + odds
│   └── error_model.json                    # Error model parameters
│
└── results/
    ├── backtest_results.csv                # Bet-by-bet results
    ├── backtest_summary.json               # Overall performance metrics
    └── strategy_optimization.csv           # Parameter optimization results
```

## Performance Considerations

### API Efficiency

- **Batch Requests**: Fetch multiple days in single API call when possible
- **Caching**: Cache API responses to avoid redundant calls
- **Parallel Requests**: Use async/concurrent requests for independent fetches
- **Rate Limiting**: Add delays between requests to be respectful

### Data Processing

- **Pandas Optimization**: Use vectorized operations instead of loops
- **Memory Management**: Process data in chunks if dataset becomes large
- **Incremental Updates**: Only fetch new data, not entire history each run

### Storage

- **CSV Format**: Use CSV for easy inspection and compatibility
- **Compression**: Compress large CSV files (gzip)
- **Indexing**: Add date index to CSVs for fast lookups

## Future Enhancements

### Phase 2 Features

1. **Multiple Forecast Sources**: Incorporate NWS, ECMWF, HRRR forecasts
2. **Ensemble Forecasting**: Blend multiple models for better accuracy
3. **Real-time Monitoring**: Track live forecast vs odds for current markets
4. **Automated Betting**: Connect to Polymarket API for actual bet placement

### Advanced Error Modeling

1. **Machine Learning**: Use XGBoost/LightGBM for error prediction
2. **Feature Engineering**: Add weather patterns, atmospheric conditions
3. **Conditional Models**: Separate models for different weather regimes
4. **Probabilistic Forecasts**: Generate full probability distributions

### Strategy Improvements

1. **Dynamic Kelly**: Adjust Kelly fraction based on recent performance
2. **Portfolio Optimization**: Optimize across multiple markets simultaneously
3. **Risk Management**: Add stop-loss and position limits
4. **Market Impact**: Model how bet size affects odds

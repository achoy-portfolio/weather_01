# NO Bet Strategy Guide

## Overview

The improved backtest strategy now includes **NO betting** on temperature ranges, which significantly improves profitability. This strategy bets NO on temperature ranges that are far from your forecast, capitalizing on the market's uncertainty.

## How It Works

### The Problem

When you forecast 25-26°F, you don't know if the actual temperature will resolve to:

- 23-24°F (below your forecast)
- 25-26°F (matching your forecast)
- 27-28°F (above your forecast)

### The Solution

Instead of only betting YES on 25-26°F, you can **also bet NO** on ranges that are far from your forecast:

- Bet NO on 21-22°F (confident it will be warmer)
- Bet NO on 23-24°F (likely too cold)
- Bet YES on 25-26°F (your forecast)
- Consider NO on 29-30°F (likely too warm)

## Strategy Parameters

### `enable_no_bets` (default: True)

Enable or disable NO betting entirely.

### `no_bet_min_distance` (default: 2°F)

Minimum distance in degrees from your forecast to consider betting NO.

Example: If forecast is 50°F and `no_bet_min_distance=2`:

- Will consider NO bets on: 48°F and below, 52°F and above
- Will NOT bet NO on: 49°F, 50°F, 51°F (too close to forecast)

## Performance Results

### Overall Statistics

- **NO Bets Placed**: 654
- **YES Bets Placed**: 149
- **NO Bet Win Rate**: 96.2%
- **YES Bet Win Rate**: 29.5%
- **NO Bet ROI**: 68.5%
- **YES Bet ROI**: 27.6%

### Profit Breakdown

- **Total Profit**: $232,135
- **NO Bets Profit**: $190,648 (82%)
- **YES Bets Profit**: $41,488 (18%)

### Performance by Distance

| Distance from Forecast | Count | Win Rate | Total Profit |
| ---------------------- | ----- | -------- | ------------ |
| 2-3°F                  | 170   | 91.2%    | $93,217      |
| 3-4°F                  | 155   | 98.1%    | $38,059      |
| 4-5°F                  | 116   | 99.1%    | $27,040      |
| 5-10°F                 | 187   | 99.5%    | $23,297      |
| 10+°F                  | 3     | 100.0%   | $221         |

**Key Insight**: The 2-3°F distance range is most profitable because:

1. Market still assigns significant probability to these ranges
2. Your model has strong edge (25.6% average)
3. High volume of opportunities

### Performance by Lead Time

**1-Day Ahead Forecasts:**

- YES Bets: 76 bets, 23.7% win rate, -$38,500 profit
- NO Bets: 323 bets, 93.8% win rate, +$61,059 profit

**Same-Day Forecasts:**

- YES Bets: 73 bets, 35.6% win rate, +$79,988 profit
- NO Bets: 331 bets, 98.5% win rate, +$129,589 profit

**Key Insight**: Same-day forecasts have higher accuracy, leading to better performance on both YES and NO bets.

## When NO Bets Lose

NO bets lose when the actual temperature is further from your forecast than expected. Examples:

1. **Forecast: 53.6°F, Actual: 51.0°F**
   - Bet NO on 51-52°F (2.1°F below forecast)
   - Lost because actual landed in that range
   - Model gave 91% chance of NO winning

2. **Forecast: 68.6°F, Actual: 66.0°F**
   - Bet NO on 66-67°F (2.1°F below forecast)
   - Lost because forecast was too high
   - Largest single loss: -$1,631

**Pattern**: Most losses occur when:

- Betting NO on ranges 2-3°F from forecast (closest distance)
- Forecast error is larger than expected
- 1-day ahead forecasts (higher uncertainty)

## Optimal Strategy

### Recommended Settings

```python
backtest_strategy(
    enable_no_bets=True,      # Enable NO betting
    no_bet_min_distance=2,    # Bet NO on ranges 2+ degrees away
    min_edge=0.05,            # Require 5% edge
    min_ev=0.05,              # Require 5% expected value
    kelly_fraction=0.25       # Use quarter Kelly for safety
)
```

### Best Practices

1. **Distance Selection**
   - 2-3°F: Highest profit but lower win rate (91%)
   - 3-5°F: Very high win rate (98-99%) with good profit
   - 5+°F: Near-perfect win rate but lower market odds

2. **Lead Time Preference**
   - Same-day forecasts: 98.5% win rate on NO bets
   - 1-day ahead: 93.8% win rate on NO bets
   - Use same-day forecasts when available

3. **Risk Management**
   - NO bets have lower variance than YES bets
   - Can size NO bets slightly larger due to higher win rate
   - Still respect Kelly criterion and liquidity constraints

4. **Market Selection**
   - Look for ranges where market assigns 20-80% probability
   - Avoid near-certain markets (>95% or <5%)
   - Ensure minimum volume ($100+)

## Example Scenarios

### Scenario 1: Cold Day Forecast

**Forecast**: 25°F

**Betting Opportunities**:

- NO on 19-20°F (5°F below) - Very safe, low odds
- NO on 21-22°F (3.5°F below) - Safe, moderate odds
- NO on 23-24°F (2°F below) - Good edge, higher odds
- YES on 25-26°F - Your forecast range
- NO on 27-28°F (2.5°F above) - Good edge
- NO on 29-30°F (4.5°F above) - Very safe

### Scenario 2: Warm Day Forecast

**Forecast**: 75°F

**Betting Opportunities**:

- NO on 69-70°F (5°F below)
- NO on 71-72°F (3.5°F below)
- NO on 73-74°F (2°F below)
- YES on 75-76°F
- NO on 77-78°F (2.5°F above)
- NO on 79-80°F (4.5°F above)

## Code Implementation

The NO betting logic is implemented in `scripts/analysis/backtest_betting_strategy.py`:

```python
def should_consider_no_bet(forecasted_max, threshold_value, threshold_type, min_distance=2):
    """
    Determine if we should consider betting NO on a threshold.

    Logic: If forecast is 25-26°F, we might bet NO on 23-24 or 21-22
    because we're confident the temp will be higher.
    """
    if threshold_type != 'range':
        return False

    distance = forecasted_max - threshold_value
    return distance >= min_distance

def calculate_no_bet_probability(forecast_temp, threshold, threshold_type, error_model):
    """
    Calculate probability that a NO bet wins.
    NO bet wins when YES bet loses.
    """
    yes_prob = calculate_model_probability(forecast_temp, threshold, threshold_type, error_model)
    return 1 - yes_prob
```

## Running the Analysis

### Run Backtest with NO Bets

```bash
python scripts/analysis/backtest_betting_strategy.py
```

### Analyze NO Bet Performance

```bash
python scripts/analysis/analyze_no_bet_strategy.py
```

### Disable NO Bets (for comparison)

```python
results = backtest_strategy(
    enable_no_bets=False  # Only bet YES
)
```

## Key Takeaways

1. **NO betting is highly profitable**: 82% of total profit comes from NO bets
2. **High win rate**: 96.2% win rate on NO bets vs 29.5% on YES bets
3. **Lower variance**: NO bets are more consistent and predictable
4. **Optimal distance**: 2-3°F from forecast provides best risk/reward
5. **Same-day advantage**: Same-day forecasts have 98.5% win rate on NO bets
6. **Diversification**: Betting both YES and NO provides better portfolio performance

## Future Improvements

1. **Dynamic distance thresholds**: Adjust `no_bet_min_distance` based on forecast uncertainty
2. **Temperature-dependent strategy**: Use different distances for cold vs warm days
3. **Market depth analysis**: Consider betting NO more aggressively on liquid markets
4. **Correlation analysis**: Avoid betting NO on adjacent ranges simultaneously
5. **Time-based adjustments**: Increase NO bet distance as event approaches (less uncertainty)

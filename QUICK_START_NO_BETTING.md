# Quick Start: NO Betting Strategy

## TL;DR

The backtest now supports betting **NO** on temperature ranges far from your forecast, increasing profit by **10x** and win rate from 30% to 84%.

## Quick Comparison

| Metric       | YES-Only | YES+NO   | Improvement |
| ------------ | -------- | -------- | ----------- |
| Bets Placed  | 149      | 803      | +654 (5.4x) |
| Win Rate     | 29.5%    | 83.8%    | +54.3%      |
| Total Profit | $22,868  | $232,135 | +$209,268   |
| ROI          | +2,287%  | +23,214% | **10.2x**   |

## How It Works

**Forecast: 50°F**

### Old Strategy (YES-only)

- Bet YES on 50-51°F
- 1 betting opportunity

### New Strategy (YES+NO)

- Bet NO on 44-45°F (6°F below)
- Bet NO on 46-47°F (4°F below)
- Bet NO on 48-49°F (2°F below)
- Bet YES on 50-51°F (your forecast)
- Bet NO on 52-53°F (2.5°F above)
- Bet NO on 54-55°F (4.5°F above)
- **5-6 betting opportunities**

## Run It Now

```bash
# Run backtest with NO bets (default)
python scripts/analysis/backtest_betting_strategy.py

# Analyze NO bet performance
python scripts/analysis/analyze_no_bet_strategy.py

# Compare strategies
python scripts/analysis/compare_yes_vs_no_strategy.py
```

## Key Settings

```python
backtest_strategy(
    enable_no_bets=True,      # Enable NO betting (default)
    no_bet_min_distance=2,    # Bet NO on ranges 2+ degrees away (default)
    min_edge=0.05,            # Require 5% edge
    min_ev=0.05               # Require 5% expected value
)
```

## Why It Works

1. **High Confidence**: You're 95%+ confident temp won't be 10°F away from forecast
2. **Market Inefficiency**: Markets still price these at 10-30% probability
3. **More Opportunities**: 4-5x more bets per day
4. **Lower Risk**: 96% win rate on NO bets

## Performance by Distance

| Distance | Win Rate | Profit                |
| -------- | -------- | --------------------- |
| 2-3°F    | 91.2%    | $93,217 (best profit) |
| 3-4°F    | 98.1%    | $38,059               |
| 4-5°F    | 99.1%    | $27,040               |
| 5-10°F   | 99.5%    | $23,297               |

**Sweet Spot**: 2-4°F from forecast

## Risk Management

### When NO Bets Lose

- Forecast error larger than expected
- Betting too close to forecast (2-3°F)
- 1-day ahead forecasts (higher uncertainty)

### Mitigation

- Use `no_bet_min_distance=3` for conservative approach
- Prefer same-day forecasts (98.5% win rate)
- Monitor forecast accuracy

## Example Results

**Sample NO Bet (Winner)**

```
Date: 2025-03-01
Forecast: 64.8°F, Actual: 64.0°F
Bet: NO on 57-58°F (7°F below forecast)
Model: 100% NO wins, Market: 52% NO wins
Edge: +48%
Result: ✅ WON $58.02
```

**Sample NO Bet (Loser)**

```
Date: 2025-06-01
Forecast: 68.6°F, Actual: 66.0°F
Bet: NO on 66-67°F (2.1°F below forecast)
Model: 87% NO wins, Market: 62.5% NO wins
Edge: +24.5%
Result: ❌ LOST -$1,630.67
```

## Documentation

- **Full Guide**: `markdown/NO_BET_STRATEGY_GUIDE.md`
- **Implementation Details**: `BACKTEST_NO_BET_IMPROVEMENT.md`
- **Code**: `scripts/analysis/backtest_betting_strategy.py`

## Disable NO Bets

To compare or disable:

```python
backtest_strategy(enable_no_bets=False)  # YES-only strategy
```

## Bottom Line

**NO betting is now the primary profit driver**, contributing 82% of total profits with a 96% win rate. Enable it by default and adjust `no_bet_min_distance` based on your risk tolerance.

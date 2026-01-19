# Betting Strategy Backtest

## Overview

This betting strategy uses weather forecast error modeling to find profitable betting opportunities on Polymarket temperature markets.

## How It Works

### 1. Error Model

- Uses historical forecast accuracy data (MAE: 2.17°F, Bias: +0.61°F, Std: 2.58°F)
- Forecasts are systematically 0.61°F too warm
- Errors follow a normal distribution

### 2. Probability Calculation

For each betting opportunity:

1. Get 9 PM forecast from day before
2. Adjust for bias: `adjusted_forecast = forecast - 0.61°F`
3. Calculate model probability using normal distribution
4. Compare to market probability to find edge

### 3. Bet Sizing

Uses Kelly Criterion with safeguards:

- Quarter Kelly (25% of full Kelly)
- Max 5% of bankroll per bet
- Only bet on markets with ≥5% market probability (avoid illiquid markets)

### 4. Bet Criteria

A bet is placed when:

- Edge ≥ 5% (model_prob - market_prob)
- Expected Value ≥ 5%
- Market probability between 5% and 95%

## Results Summary

**Overall Performance:**

- 68 bets placed out of 2,156 opportunities
- 23.5% win rate
- +172.7% ROI on starting bankroll
- +34.8% ROI on total wagered

**Key Findings:**

1. **"Above" bets are profitable**
   - 43.3% win rate
   - +$1,987 total profit
   - Model is good at predicting warm temperatures

2. **"Below" bets lose money**
   - 0% win rate (!)
   - -$1,140 total loss
   - Model systematically fails on cold predictions

3. **High edge bets (>30%) are traps**
   - Only 22.2% win rate
   - -$50.80 average profit
   - Model is overconfident on extreme edges

4. **Moderate edge bets (10-20%) perform best**
   - 33.3% win rate
   - +$68.68 average profit

## Files

- `scripts/analysis/backtest_betting_strategy.py` - Main backtest script
- `scripts/analysis/betting_simulator.py` - Results analysis
- `data/results/backtest_results.csv` - Full results (all opportunities)
- `data/results/betting_opportunities.csv` - Only bets that were placed
- `data/results/betting_outcomes.csv` - Win/loss outcomes

## Usage

```bash
# Run backtest with default parameters
python scripts/analysis/backtest_betting_strategy.py

# Analyze results
python scripts/analysis/betting_simulator.py
```

## Strategy Parameters

You can adjust these in `backtest_betting_strategy.py`:

```python
results = backtest_strategy(
    min_edge=0.05,          # Minimum 5% edge required
    min_ev=0.05,            # Minimum 5% expected value
    min_market_prob=0.05,   # Avoid markets below 5% probability
    bankroll=1000,          # Starting bankroll
    kelly_fraction=0.25,    # Use quarter Kelly
    max_bet_pct=0.05        # Max 5% of bankroll per bet
)
```

## Recommendations

Based on the backtest results:

1. **Focus on "above" bets** - they have 43% win rate vs 0% for "below"
2. **Avoid high edge opportunities (>30%)** - likely model overconfidence
3. **Target moderate edges (10-20%)** - best risk/reward
4. **Be cautious in winter months** - model struggles with cold temperatures
5. **June-August are best** - model is most accurate for summer temps

## Limitations

1. **Survivorship bias** - only includes days with odds data
2. **Liquidity assumptions** - assumes we can always get the quoted odds
3. **Market efficiency** - real markets may be more efficient than historical data suggests
4. **Sample size** - only 68 bets, need more data for statistical significance
5. **Cold weather bias** - model fails on "below" bets, needs investigation

## Next Steps

1. Investigate why "below" bets have 0% win rate
2. Build separate error models for warm vs cold predictions
3. Add seasonal adjustments to the model
4. Test with different Kelly fractions and bet size limits
5. Implement live tracking to validate strategy in real-time

# Betting Recommendation Dashboard Guide

## Overview

The betting recommendation dashboard provides real-time betting recommendations based on Open-Meteo forecasts and the proven backtest strategy.

## Features

### 1. Real-Time Forecast Integration

- Fetches Open-Meteo forecasts for up to 16 days ahead
- Calculates forecasted maximum temperature for the target date
- Uses same forecast source as historical analysis

### 2. Live Market Odds

- Fetches current Polymarket odds for all temperature ranges
- Shows market probability, volume, and liquidity for each threshold
- Automatically parses "above", "below", and "range" markets

### 3. Betting Recommendations

- Uses proven error models (1-day or same-day lead time)
- Calculates model probability vs market probability
- Identifies positive edge opportunities
- Applies Kelly criterion for bet sizing
- Enforces liquidity constraints (max 10% of market volume)

### 4. Customizable Parameters

- **Bankroll:** Set your available capital ($100 - $100,000)
- **Minimum Edge:** Required advantage (default 5%)
- **Kelly Fraction:** Bet sizing aggressiveness (default 25%)
- **Max Bet %:** Maximum bet as % of bankroll (default 5%)
- **Lead Time:** Choose 0-day (same-day) or 1-day ahead error model

## How to Use

### Running the Dashboard

```bash
streamlit run betting_recommendation_dashboard.py
```

The dashboard will open in your browser at `http://localhost:8501`

### Step-by-Step Guide

1. **Select Market Date**
   - Choose which date's market to analyze
   - Markets typically open 2 days before resolution
   - Default is tomorrow

2. **Set Your Bankroll**
   - Enter how much money you have available to bet
   - This affects bet sizing calculations

3. **Adjust Strategy Parameters** (optional)
   - Increase minimum edge for more conservative strategy
   - Reduce Kelly fraction for smaller bets
   - Choose lead time based on when you're betting

4. **Review Recommendations**
   - Green cards show recommended bets
   - Each card shows:
     - Bet size (with liquidity cap if applicable)
     - Edge (model prob - market prob)
     - Expected value
     - Potential profit/loss
   - Gray card means no bets meet criteria

5. **Analyze All Markets**
   - View comparison chart of model vs market probabilities
   - Review detailed table with all thresholds
   - Identify where model disagrees most with market

## Understanding the Recommendations

### Bet Card Information

```
🎲 BET: ≥32
Bet Size: $45.23 [LIQUIDITY CAPPED]
Edge: +12.5%
Model Probability: 65.3%
Market Probability: 52.8%
Expected Value: +23.7%
Market Volume: $8,450
Potential Profit: $41.23 if win | Loss: $45.23 if lose
```

- **Bet Size:** How much to wager (capped by liquidity if noted)
- **Edge:** Your advantage (positive = good opportunity)
- **Model Probability:** What the forecast suggests
- **Market Probability:** What the market thinks
- **Expected Value:** Expected return per dollar wagered
- **Market Volume:** Total money in the market
- **Potential Profit:** Upside if you win vs downside if you lose

### When to Bet

✅ **Good Opportunities:**

- High edge (>10%)
- High expected value (>10%)
- Reasonable market volume (>$1,000)
- Model probability significantly different from market

❌ **Avoid:**

- Low edge (<5%)
- Very low volume markets (<$100)
- Markets where you're liquidity constrained
- When forecast is uncertain (check std dev)

## Error Models

### Same-Day (0d)

- **Bias:** +0.26°F
- **Std Dev:** 1.60°F
- **MAE:** 1.10°F
- **Use when:** Betting on the actual day (most accurate)

### 1-Day Ahead (1d)

- **Bias:** -0.49°F
- **Std Dev:** 3.13°F
- **MAE:** 2.40°F
- **Use when:** Betting the day before (less accurate but still good)

## Strategy Parameters Explained

### Minimum Edge

- **Conservative:** 10% (fewer bets, higher quality)
- **Moderate:** 5% (balanced approach)
- **Aggressive:** 3% (more bets, lower quality)

### Kelly Fraction

- **Conservative:** 10% (1/10th Kelly, very safe)
- **Moderate:** 25% (1/4 Kelly, recommended)
- **Aggressive:** 50% (1/2 Kelly, higher variance)

### Max Bet %

- **Conservative:** 2% (limits single bet risk)
- **Moderate:** 5% (balanced)
- **Aggressive:** 10% (higher risk per bet)

## Liquidity Constraints

The dashboard enforces realistic liquidity limits:

- **Minimum Volume:** $100 (filters out illiquid markets)
- **Max Bet vs Volume:** 10% (can't bet more than 10% of market volume)
- **Effect:** Prevents unrealistic bet sizes that would move the market

When a bet is liquidity constrained, you'll see `[LIQUIDITY CAPPED]` next to the bet size.

## Example Scenarios

### Scenario 1: Strong Recommendation

```
Forecast Max: 32.5°F
Market: ≥32°F at 45% probability
Model: ≥32°F at 68% probability
Edge: +23%
Recommendation: BET $42.50
```

**Why:** Large edge, model strongly disagrees with market

### Scenario 2: No Recommendation

```
Forecast Max: 32.5°F
Market: ≥32°F at 65% probability
Model: ≥32°F at 68% probability
Edge: +3%
Recommendation: NO BET
```

**Why:** Edge too small (below 5% minimum)

### Scenario 3: Liquidity Constrained

```
Forecast Max: 32.5°F
Market: ≥32°F at 25% probability (Volume: $500)
Model: ≥32°F at 75% probability
Edge: +50%
Kelly suggests: $150
Recommendation: BET $50 [LIQUIDITY CAPPED]
```

**Why:** Can only bet 10% of $500 volume = $50

## Tips for Success

1. **Start Small:** Begin with a small bankroll to test the strategy
2. **Track Results:** Keep a log of your bets and outcomes
3. **Be Patient:** Not every day will have good opportunities
4. **Check Forecast Quality:** Look at the std dev - higher = more uncertain
5. **Monitor Volume:** Higher volume markets are more liquid and reliable
6. **Use Conservative Settings:** Start with higher minimum edge and lower Kelly fraction
7. **Don't Chase Losses:** Stick to the strategy even after losses
8. **Understand Variance:** Even with positive edge, you'll have losing streaks

## Limitations & Risks

⚠️ **Important Disclaimers:**

1. **Model Uncertainty:** Forecasts can be wrong, especially far in advance
2. **Market Efficiency:** Markets may know something the model doesn't
3. **Slippage:** Actual execution prices may differ from quoted odds
4. **Fees:** Polymarket charges fees on winnings
5. **Liquidity:** Small markets may not fill your full bet size
6. **Variance:** High edge doesn't guarantee wins on individual bets
7. **Past Performance:** Backtest results don't guarantee future success

## Troubleshooting

### "No forecast data available"

- Date may be too far in future (>16 days)
- Date may be in the past
- Open-Meteo API may be down
- Try refreshing or selecting a different date

### "No Polymarket market found"

- Market may not be open yet (opens ~2 days before)
- Date format may be wrong
- Market may have closed
- Try tomorrow or the next day

### "No Bets Recommended"

- Markets may be fairly priced
- Edge may be below minimum threshold
- Try lowering minimum edge parameter
- Check if forecast is very uncertain

## Technical Details

### Data Sources

- **Forecast:** Open-Meteo API (free, no key required)
- **Odds:** Polymarket Gamma API (public)
- **Error Model:** Historical analysis from `check_error_distribution.py`

### Calculation Method

1. Fetch Open-Meteo hourly forecast for target date
2. Calculate forecasted maximum temperature
3. Fetch current Polymarket odds for all thresholds
4. For each threshold:
   - Calculate model probability using error distribution
   - Compare to market probability
   - Calculate edge and expected value
   - Apply Kelly criterion for bet sizing
   - Enforce liquidity constraints
5. Recommend bets that meet all criteria

### Files

- `betting_recommendation_dashboard.py` - Main dashboard
- `data/processed/error_distribution_analysis.json` - Error models
- `scripts/analysis/backtest_betting_strategy.py` - Strategy logic

## Support

For issues or questions:

1. Check error messages in the dashboard
2. Review this guide
3. Check the backtest results to understand strategy performance
4. Verify error model file exists: `data/processed/error_distribution_analysis.json`

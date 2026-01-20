# Betting Dashboard - NO Betting Update

## What's New

The betting recommendation dashboard now includes **NO betting recommendations**, showing you when to bet against temperature ranges that are far from your forecast.

## Key Features

### 1. NO Betting Toggle

- **Enable/Disable NO Bets**: Control whether to show NO betting opportunities
- **Min Distance Setting**: Adjust how far from forecast to consider NO bets (default: 2°F)

### 2. Visual Distinction

- **YES Bets**: Purple gradient cards (bet that temp WILL be in range)
- **NO Bets**: Green gradient cards (bet that temp will NOT be in range)
- Clear labeling and distance indicators

### 3. Smart Recommendations

- Shows distance from forecast for each NO bet
- Explains why NO bets make sense
- Displays win probability for NO bets (typically 95%+)

## How to Use

### Step 1: Open Dashboard

```bash
streamlit run betting_recommendation_dashboard.py
```

### Step 2: Configure Settings (Sidebar)

**Basic Settings:**

- Select market date
- Set your bankroll
- Adjust minimum edge (5% recommended)
- Set Kelly fraction (25% recommended)

**NO Betting Settings:**

- ✅ Enable NO Bets (checked by default)
- Set Min Distance: 2°F (recommended)
  - 2°F: More aggressive, more opportunities
  - 3°F: More conservative, higher win rate

### Step 3: Review Recommendations

The dashboard shows:

1. **Summary Stats**
   - Total bets recommended
   - YES bets count
   - NO bets count

2. **YES Bets Section** (Purple cards)
   - Bet that temperature WILL be in the range
   - Shows edge, EV, bet size
   - Potential profit/loss

3. **NO Bets Section** (Green cards)
   - Bet that temperature will NOT be in the range
   - Shows distance from forecast
   - Explains why NO bet makes sense
   - Typically 95%+ win probability

## Example Scenario

**Forecast**: 50°F for January 20th

**Dashboard Shows:**

### YES Bets

- ✅ **BET YES on 50-51°F**
  - Model: 35%, Market: 45%
  - Edge: -10% (SKIP - negative edge)

### NO Bets

- 🚫 **BET NO on 44-45°F**
  - 5.5°F below forecast
  - Model: 98% NO wins, Market: 85% NO wins
  - Edge: +13%
  - Bet: $45.00

- 🚫 **BET NO on 46-47°F**
  - 3.5°F below forecast
  - Model: 95% NO wins, Market: 75% NO wins
  - Edge: +20%
  - Bet: $52.00

- 🚫 **BET NO on 54-55°F**
  - 4.5°F above forecast
  - Model: 97% NO wins, Market: 80% NO wins
  - Edge: +17%
  - Bet: $48.00

**Result**: 3 profitable NO bets instead of 0 YES bets!

## Understanding NO Bets

### Why Bet NO?

When your forecast is 50°F:

- You're very confident it won't be 44°F (6°F below)
- You're very confident it won't be 56°F (6°F above)
- Market still assigns 10-20% probability to these ranges
- This creates profitable opportunities

### Win Rate

Based on backtest results:

- **2-3°F distance**: 91% win rate
- **3-4°F distance**: 98% win rate
- **4-5°F distance**: 99% win rate
- **5+°F distance**: 99.5% win rate

### Risk Considerations

NO bets lose when:

- Forecast error is larger than expected
- Betting too close to forecast (2-3°F)
- Using 1-day ahead forecasts (higher uncertainty)

**Mitigation**:

- Use same-day forecasts when possible (98.5% win rate)
- Increase min distance to 3°F for conservative approach
- Monitor forecast accuracy

## Settings Guide

### Recommended Settings

**Aggressive Strategy** (More bets, higher variance):

```
Min Edge: 5%
Kelly Fraction: 25%
Max Bet: 5%
Enable NO Bets: ✅
NO Bet Min Distance: 2°F
```

**Conservative Strategy** (Fewer bets, lower variance):

```
Min Edge: 10%
Kelly Fraction: 20%
Max Bet: 3%
Enable NO Bets: ✅
NO Bet Min Distance: 3°F
```

**YES-Only Strategy** (Original behavior):

```
Min Edge: 5%
Kelly Fraction: 25%
Max Bet: 5%
Enable NO Bets: ❌
```

## Visual Guide

### YES Bet Card (Purple)

```
🎲 BET YES: 50-51°F
Bet Size: $45.00
Edge: +8%
Model Probability: 35%
Market Probability: 27%
```

### NO Bet Card (Green)

```
🚫 BET NO: 44-45°F
5.5°F below your forecast of 50.0°F

Bet Size: $45.00
Edge: +13%
Model Prob (NO wins): 98%
Market Prob (NO wins): 85%
```

## Data Table

The detailed analysis table now includes:

- **Side**: YES or NO
- **Range**: Temperature range
- **Model Prob**: Your model's probability
- **Market Prob**: Market's probability
- **Edge**: Your advantage
- **EV**: Expected value
- **Volume**: Market liquidity
- **Bet?**: Recommendation
- **Bet Size**: Suggested bet amount
- **Distance**: Distance from forecast (for NO bets)

## Performance Expectations

Based on historical backtest:

### YES Bets

- Win Rate: ~30%
- Average Edge: 20%
- ROI: 28%

### NO Bets

- Win Rate: ~96%
- Average Edge: 22%
- ROI: 69%

### Combined Strategy

- Total Bets: 5-6x more opportunities
- Overall Win Rate: ~84%
- Total ROI: Significantly higher

## Tips for Success

1. **Start Conservative**: Use 3°F min distance initially
2. **Monitor Results**: Track your actual win rate vs expected
3. **Prefer Same-Day**: Same-day forecasts have 98.5% NO bet win rate
4. **Check Liquidity**: Ensure sufficient market volume
5. **Diversify**: Bet both YES and NO for better portfolio performance
6. **Respect Kelly**: Don't exceed recommended bet sizes
7. **Review Distance**: Most profit comes from 2-4°F distance

## Troubleshooting

### No NO Bets Showing

- Check "Enable NO Bets" is checked
- Reduce "NO Bet Min Distance" to 2°F
- Ensure forecast is available
- Check that markets exist far from forecast

### Too Many NO Bets

- Increase "NO Bet Min Distance" to 3-4°F
- Increase "Min Edge" to 10%
- This will filter to only highest-confidence bets

### Low Win Rate on NO Bets

- Increase "NO Bet Min Distance" to 3°F
- Use same-day forecasts instead of 1-day ahead
- Check forecast accuracy in your region

## Next Steps

1. Run the dashboard: `streamlit run betting_recommendation_dashboard.py`
2. Select tomorrow's date
3. Review both YES and NO recommendations
4. Start with small bets to validate strategy
5. Track results and adjust settings
6. Scale up as confidence grows

## Related Documentation

- **Strategy Guide**: `markdown/NO_BET_STRATEGY_GUIDE.md`
- **Backtest Results**: `BACKTEST_NO_BET_IMPROVEMENT.md`
- **Quick Start**: `QUICK_START_NO_BETTING.md`
- **Analysis Tools**: `scripts/analysis/analyze_no_bet_strategy.py`

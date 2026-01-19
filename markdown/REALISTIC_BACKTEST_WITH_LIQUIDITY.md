# Realistic Backtest with Liquidity Constraints

## Summary

Updated betting strategy to use 1-day and same-day forecasts with realistic liquidity constraints based on market volume.

## Key Changes

### 1. Lead Time Strategy

**Changed from:** 2-day lead time forecasts
**Changed to:** 1-day and same-day (0d) forecasts

**Rationale:** Closer forecasts are more accurate, leading to better edge identification.

### 2. Liquidity Constraints Added

- **Minimum Volume:** $100 (filters out illiquid markets)
- **Max Bet vs Volume:** 10% (can't bet more than 10% of market volume)
- **Effect:** 22.1% of bets were liquidity constrained

### 3. Error Models Used

- **1-day ahead:** Bias -0.49°F, Std 3.13°F, MAE 2.40°F
- **Same-day:** Bias +0.26°F, Std 1.60°F, MAE 1.10°F

## Results Comparison

### Without Liquidity Constraints

- ROI: +3,608%
- Ending Bankroll: $37,082
- Avg Bet Size: $738
- Unrealistic due to unlimited bet sizes

### With Liquidity Constraints (REALISTIC)

- ROI: +2,287%
- Ending Bankroll: $23,868
- Avg Bet Size: $473
- Liquidity Constrained: 22.1% of bets

## Detailed Results

### Overall Performance

- Total Opportunities: 4,312
- Bets Placed: 149
- Bets Won: 44
- Win Rate: 29.5%
- Total Wagered: $70,447
- Total Profit: +$22,868
- ROI on Wagered: +32.5%

### Market Volume Statistics

- Mean Volume: $17,058
- Median Volume: $10,455
- Min Volume: $552
- Max Volume: $125,635

### Bet Size Statistics

- Mean: $473
- Median: $346
- Min: $17
- Max: $1,824 (liquidity capped)

### Performance by Edge

- 5-10% edge: 60 bets, 21.7% win rate, -$20.61 avg profit
- 10-20% edge: 37 bets, 27.0% win rate, -$56.18 avg profit
- 20-30% edge: 13 bets, 15.4% win rate, -$512.14 avg profit
- **>30% edge: 39 bets, 48.7% win rate, +$842.08 avg profit** ⭐

### Performance by Threshold Type

- **Above (≥):** 68 bets, 44.1% win rate, +$18,919 profit ⭐
- Below (≤): 15 bets, 13.3% win rate, +$1,135 profit
- Range (X-Y): 66 bets, 18.2% win rate, +$2,814 profit

### Monthly Performance

| Month   | Bets | Win Rate | Profit      |
| ------- | ---- | -------- | ----------- |
| 2025-02 | 1    | 100.0%   | +$72        |
| 2025-03 | 16   | 43.8%    | +$3,871     |
| 2025-04 | 12   | 16.7%    | -$279       |
| 2025-05 | 18   | 22.2%    | +$2,166     |
| 2025-06 | 19   | 31.6%    | +$2,082     |
| 2025-07 | 13   | 69.2%    | +$5,146     |
| 2025-08 | 17   | 17.6%    | +$6,858     |
| 2025-09 | 15   | 40.0%    | +$21,557 ⭐ |
| 2025-10 | 11   | 9.1%     | -$6,404     |
| 2025-11 | 10   | 10.0%    | -$8,150     |
| 2025-12 | 14   | 28.6%    | -$2,289     |
| 2026-01 | 3    | 0.0%     | -$1,763     |

## Best Bets (Top 5)

1. **2025-09-18 (0d): ≥83°F** - Edge 81.1%, Bet $950, Profit +$15,862
2. **2025-08-20 (0d): ≤70°F** - Edge 45.7%, Bet $661, Profit +$10,733
3. **2025-12-01 (0d): 43-44°F** - Edge 10.6%, Bet $853, Profit +$5,709
4. **2025-09-22 (0d): 71-72°F** - Edge 9.0%, Bet $671, Profit +$5,428
5. **2025-09-18 (1d): ≥83°F** - Edge 53.6%, Bet $1,392, Profit +$4,782

## Key Insights

### What Works

1. **High edge bets (>30%)** have 48.7% win rate and strong profitability
2. **"Above" threshold bets** significantly outperform other types
3. **Same-day forecasts** provide better accuracy for final betting decisions
4. **Summer months** (July-September) showed strongest performance

### Concerns & Limitations

1. **Low overall win rate (29.5%)** - relies on high payouts when winning
2. **High variance** - September made $21k, but Nov/Dec lost $10k+
3. **Liquidity constraints** - 22% of bets were capped by market volume
4. **Kelly criterion compounding** - bet sizes grow exponentially with bankroll
5. **Slippage not modeled** - actual execution may be worse than backtest prices
6. **Market impact not modeled** - large bets may move prices against you

### Realism Factors Still Missing

- **Slippage:** Actual fill prices may be worse than quoted odds
- **Market impact:** Your bet may move the market against you
- **Timing:** Odds change rapidly; may not get desired price
- **Fees:** Polymarket charges fees on winnings
- **Withdrawal limits:** May not be able to access full bankroll
- **Market closure:** Some markets may close before you can bet

## Strategy Parameters

```python
lead_times = ['1d', '0d']      # 1-day and same-day forecasts
min_edge = 0.05                # 5% minimum edge
min_ev = 0.05                  # 5% minimum expected value
min_market_prob = 0.05         # 5% minimum market probability
min_volume = 100               # $100 minimum volume
max_bet_vs_volume = 0.10       # Max 10% of market volume
bankroll = 1000                # $1,000 starting bankroll
kelly_fraction = 0.25          # Quarter Kelly sizing
max_bet_pct = 0.05             # Max 5% of bankroll per bet
```

## Recommendations

### For More Conservative Strategy

- Increase `min_edge` to 0.10 (10%)
- Reduce `kelly_fraction` to 0.10 (1/10th Kelly)
- Reduce `max_bet_pct` to 0.02 (2% of bankroll)
- Increase `min_volume` to 1000 ($1,000)
- Focus only on "above" threshold bets

### For More Realistic Modeling

- Add slippage factor (e.g., 1-2% worse odds)
- Model market impact for large bets
- Add transaction fees
- Consider time-of-day effects on liquidity
- Model odds movement between forecast and bet placement

### For Risk Management

- Set maximum drawdown limits
- Implement stop-loss rules
- Diversify across multiple markets
- Reserve portion of bankroll for high-conviction bets
- Track real-time performance vs backtest expectations

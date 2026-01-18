# January 17 Odds Analysis: Why 40-41°F Odds Increased Despite 39.9°F Peak

## The Mystery

- **Peak temperature**: 39.9°F at 9:51 AM
- **40-41°F odds at 10:00 AM**: 96.0%
- **Question**: Why did traders bet so heavily on 40-41°F when temperature never reached 40°F?

## Timeline Analysis

### Temperature Progression

```
08:00 AM: 37.4°F
08:10 AM: 39.2°F  ← Jumped to 39.2°F
08:10-09:50 AM: Stayed at 39.2°F (40 minutes!)
09:51 AM: 39.9°F  ← Peak (only 0.1°F below 40°F!)
10:00 AM: 39.2°F  ← Dropped back down
10:25 AM: 37.4°F  ← Cooling trend
```

### Market Odds for 40-41°F

```
08:40 AM: 75.0%   ← Temp was 39.2°F
09:00 AM: 69.0%   ← Temp still 39.2°F
09:40 AM: 88.0%   ← Temp still 39.2°F
10:00 AM: 96.0%   ← Temp at 39.2°F (peak was 9 min ago)
10:40 AM: 97.5%   ← Temp dropped to 37.4°F
11:00 AM: 98.5%   ← Temp at 35.6°F
```

## Why Traders Were Confident About 40-41°F

### 1. **Measurement Precision**

- NWS reports to **whole degrees** (39°F, 40°F, 41°F)
- The actual reading of 39.9°F gets **rounded to 40°F** in official reports
- Traders knew: 39.9°F = 40°F for resolution purposes

### 2. **Wunderground Resolution Source**

From Polymarket rules:

> "The resolution source for this market measures temperatures to whole degrees Fahrenheit (eg, 21°F). Thus, this is the level of precision that will be used when resolving the market."

So:

- 39.5°F - 40.4°F → Resolves as **40°F**
- 40.5°F - 41.4°F → Resolves as **41°F**

### 3. **Information Advantage**

Traders likely:

- Saw the 39.9°F reading immediately (via API or faster sources)
- Knew it would round to 40°F
- Understood the market would resolve to 40-41°F bucket
- Bet heavily before others realized

### 4. **Market Timing**

Notice the odds progression:

- **Before 9:51 AM**: Odds were 70-88% (uncertain)
- **After 9:51 AM**: Odds jumped to 96%+ (very confident)
- **After 10:00 AM**: Odds stayed 96-98% (locked in)

This suggests traders saw the 39.9°F reading and immediately understood it meant 40°F for resolution.

## The Answer to Your Question

> "Do other traders have models to predict better than actually reported?"

**Not exactly.** They don't predict better - they understand the **resolution rules** better:

1. **Rounding rules**: 39.9°F rounds to 40°F
2. **Data access**: Some traders may have faster access to NWS data
3. **Market mechanics**: They understood that 39.9°F = guaranteed win for 40-41°F bucket
4. **Arbitrage opportunity**: When odds were 75% but temperature hit 39.9°F, smart traders knew it was essentially 100% and bet heavily

## Key Lessons

### For Betting

1. **Read resolution rules carefully** - rounding matters!
2. **Understand data sources** - Wunderground rounds to whole degrees
3. **Act fast** - Information edge disappears quickly
4. **Watch for precision differences** - NWS shows 39.9°F, but market resolves at 40°F

### For Your Models

1. **Account for rounding** in your predictions
2. **Use the same precision** as the resolution source
3. **Consider that 39.5°F+ is effectively 40°F** for these markets
4. **Factor in measurement uncertainty** - sensors aren't perfectly accurate

## The Real Edge

The traders who bet heavily at 10:00 AM (when odds were 96%) likely:

- Saw the 39.9°F reading at 9:51 AM
- Knew it would round to 40°F
- Understood the market would resolve to 40-41°F
- Had 9 minutes of information advantage before odds fully adjusted

This is **information arbitrage**, not prediction. They didn't predict the temperature better - they just understood the rules better and acted faster.

## Verification

Check Wunderground's historical data for Jan 17:
https://www.wunderground.com/history/daily/us/ny/new-york-city/KLGA

You'll likely see the high temperature listed as **40°F** (rounded from 39.9°F), confirming the 40-41°F bucket should win.

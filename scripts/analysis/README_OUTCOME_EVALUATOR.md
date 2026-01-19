# Outcome Evaluator

The outcome evaluator determines betting outcomes based on actual temperature observations and calculates profit/loss for each bet.

## Features

### 1. Polymarket Rounding Rules

- Applies standard rounding (round half up) to match Polymarket's whole degree resolution
- Example: 34.5°F rounds to 35°F, 34.4°F rounds to 34°F

### 2. Bet Outcome Evaluation

Evaluates whether bets would have won or lost based on actual temperatures:

- **Above threshold** (e.g., "≥75°F"): Wins if rounded actual temp >= threshold
- **Below threshold** (e.g., "≤33°F"): Wins if rounded actual temp <= threshold
- **Range threshold** (e.g., "36-37°F"): Wins if rounded actual temp is within range

### 3. Profit/Loss Calculation

- **Winning bet**: Profit = bet_size × (1/market_odds - 1)
- **Losing bet**: Loss = -bet_size

### 4. Cumulative P&L Tracking

Tracks cumulative profit/loss across all bets to show portfolio performance over time.

## Usage

### As a Module

```python
from betting_simulator import OutcomeEvaluator
import pandas as pd

# Load data
betting_decisions = [...]  # List of betting decision dicts
actual_temps_df = pd.read_csv('data/raw/actual_temperatures.csv')

# Evaluate outcomes
evaluator = OutcomeEvaluator()
results_df = evaluator.evaluate_betting_results(
    betting_decisions,
    actual_temps_df,
    starting_bankroll=1000.0
)

# Generate summary statistics
summary = evaluator.generate_summary_statistics(results_df)
print(f"Win Rate: {summary['win_rate']:.1%}")
print(f"ROI: {summary['roi']:.1f}%")
```

### As a Script

```bash
python scripts/analysis/evaluate_outcomes.py
```

## Output Format

The evaluator generates a CSV file with the following columns:

- `target_date`: Date of the bet
- `threshold`: Temperature threshold (e.g., "≥75", "36-37")
- `threshold_type`: Type of threshold ("above", "below", "range")
- `forecast_temp`: Forecasted temperature in °F
- `actual_temp`: Actual observed temperature in °F
- `bet_placed`: Whether a bet was placed (True/False)
- `bet_size`: Amount bet in dollars
- `model_probability`: Model's estimated win probability
- `market_odds`: Market's implied probability
- `expected_value`: Expected value of the bet
- `bet_outcome`: Outcome ("win", "loss", or "no_bet")
- `profit_loss`: Profit (positive) or loss (negative) in dollars
- `cumulative_pl`: Cumulative profit/loss across all bets

## Testing

Run the test suite to verify all functionality:

```bash
python scripts/analysis/test_betting_simulator.py
```

Tests cover:

- Polymarket rounding rules
- Bet outcome evaluation for all threshold types
- Profit/loss calculations
- Full betting results evaluation
- Summary statistics generation

## Requirements

Satisfies requirements 4.3 and 4.5 from the design document:

- 4.3: Compare forecast to actual outcome and record win/loss
- 4.5: Generate simulated betting record with outcomes

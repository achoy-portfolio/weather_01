# Implementation Plan

- [x] 1. Create historical forecast fetcher
  - Implement function to fetch Open-Meteo historical forecasts for a specific date and time (9 PM Eastern)
  - Add batch fetching capability for date ranges (Jan 21, 2025 onward)
  - Implement timezone handling to ensure 9 PM Eastern Time is correctly specified
  - Add retry logic with exponential backoff for API failures
  - Parse API responses to extract forecasted peak temperature
  - Save fetched forecasts to `data/raw/historical_forecasts.csv` with columns: forecast_date, forecast_time, target_date, forecasted_high, source
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [x] 2. Create actual temperature fetcher
  - Implement function to fetch actual hourly (or 5 minute or 15 minute if available)temperatures from Open-Meteo Archive API
  - Extract daily maximum, minimum, and hourly temperatures from API response
  - Calculate daily average temperature from hourly data
  - Add batch fetching for date ranges (Jan 21, 2025 onward)
  - Handle missing data gracefully with logging
  - Save actual temperatures to `data/raw/actual_temperatures.csv` with columns: date, actual_high, actual_low, actual_average, peak_time
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

- [x] 3. Enhance Polymarket historical odds fetcher
  - Fetch all odds for each betting day
  - Add function to generate event slugs for date range (Jan 22, 2025 - Jan 10, 2026)
  - Parse threshold buckets from market questions (handle "≥", "≤", and range formats)
  - Extract YES probability for each threshold from API response
  - Save odds to `data/raw/polymarket_odds_history.csv` with columns: event_date, fetch_timestamp, threshold, threshold_type, yes_probability, volume
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

- [x] 4. Create data merger and validator
  - Implement function to merge forecast, actual, and odds data by date
  - Validate data completeness (check for missing forecasts, actuals, or odds)
  - Implement outlier detection for temperatures (flag values >20°F from recent average)
  - Validate temperature bounds (-20°F to 120°F for NYC)
  - Verify consistency (high >= low, average between high and low)
  - Save merged data to `data/processed/backtest_data_combined.csv`
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6_

- [x] 5. Implement betting decision simulator
  - Use polymarket_odds_history, actual_temperatures, and historical_forecasts (read the first 100 rows of each to understand structure)
  - Create function to calculate model probability using normal distribution (z-score method)
  - Implement expected value calculation (model_prob × payout_multiplier - 1)
  - Add Kelly Criterion bet sizing with fractional Kelly (25%)
  - Implement decision logic (only bet if EV > 5%)
  - Handle different threshold types (above, below, range)
  - Create function to simulate betting decisions for all thresholds on a given day
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

- [-] 6. Implement outcome evaluator
  - Create function to determine bet outcome (win/loss) based on actual temperature
  - Implement Polymarket rounding rules (whole degree resolution)
  - Handle different threshold types in outcome evaluation
  - Calculate profit/loss for each bet
  - Track cumulative P&L across all bets
  - Generate bet-by-bet results with columns: target_date, threshold, bet_placed, bet_size, bet_outcome, profit_loss, cumulative_pl
  - _Requirements: 4.3, 4.5_

- [ ] 7. Build error model calculator
  - Implement forecast error calculation (forecast - actual)
  - Calculate Mean Absolute Error (MAE) across all forecasts
  - Calculate Root Mean Squared Error (RMSE)
  - Calculate bias (mean error) to detect systematic over/under prediction
  - Calculate standard deviation of errors
  - Break down error metrics by lead time (1-day, 2-day, etc.)
  - Break down error metrics by season (winter, spring, summer, fall)
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6_

- [ ] 8. Implement uncertainty estimator
  - Create function to estimate forecast uncertainty based on historical errors
  - Implement uncertainty adjustment for lead time (longer lead = more uncertainty)
  - Implement seasonal adjustment factors (winter more variable)
  - Add recent performance adjustment using rolling 30-day window
  - Create function to calculate confidence intervals using uncertainty estimates
  - Save error model parameters to `data/processed/error_model.json`
  - _Requirements: 5.5, 5.6, 6.1_

- [ ] 9. Integrate error model with betting strategy
  - Modify betting simulator to use error-adjusted uncertainty in probability calculations
  - Implement function to recalculate expected value using error model
  - Add confidence-based filtering (only bet when uncertainty is below threshold)
  - Create comparison report showing naive strategy vs error-adjusted strategy
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

- [ ] 10. Create backtest pipeline orchestrator
  - Implement main pipeline script that runs all steps in sequence
  - Add command-line arguments for date range, bankroll, and strategy parameters
  - Implement progress tracking and logging for each pipeline stage
  - Add error handling to continue processing if individual dates fail
  - Create checkpoint system to resume from last successful step
  - _Requirements: 1.4, 7.1, 7.2, 7.3, 7.4_

- [ ] 11. Generate performance summary reports
  - Calculate overall win rate across all bets
  - Calculate win rate by threshold type (above, below, range)
  - Calculate average forecast error by lead time
  - Calculate total simulated profit/loss
  - Calculate ROI percentage
  - Display error model statistics (MAE, RMSE, bias)
  - Save summary to `data/results/backtest_summary.json`
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

- [ ] 12. Create strategy optimizer
  - Implement function to test different EV thresholds (3%, 5%, 7%, 10%)
  - Test different Kelly fractions (10%, 25%, 50%)
  - Run backtest with each parameter combination
  - Calculate ROI and Sharpe ratio for each strategy variant
  - Identify optimal parameters based on risk-adjusted returns
  - Save optimization results to `data/results/strategy_optimization.csv`
  - _Requirements: 6.2, 6.3, 6.4, 6.5_

- [ ] 13. Add visualization and reporting utilities
  - Create function to plot forecast error distribution
  - Create function to plot cumulative P&L over time
  - Create function to plot win rate by threshold type
  - Create function to plot error metrics by lead time and season
  - Generate summary dashboard showing key metrics
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

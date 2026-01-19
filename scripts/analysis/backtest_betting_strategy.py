"""
Backtest Betting Strategy

For each betting day with Polymarket odds:
1. Get 9 PM forecast from day before
2. Calculate model probabilities for each threshold
3. Find opportunities with positive edge (model prob > market prob)
4. Simulate bets and track profit/loss
"""

import pandas as pd
import numpy as np
from scipy import stats
import json
from datetime import timedelta


def load_error_model(lead_time='1d'):
    """
    Load the saved error distribution model for specified lead time.
    
    Args:
        lead_time: '0d' or '1d' for same-day or 1-day ahead forecasts
    """
    with open('data/processed/error_distribution_analysis.json', 'r') as f:
        model = json.load(f)
    
    # Use specified lead time model
    if 'by_lead_time' in model and lead_time in model['by_lead_time']:
        return model['by_lead_time'][lead_time]
    else:
        # Fallback to overall model
        return model.get('overall', model)


def calculate_model_probability(forecast_temp, threshold, threshold_type, error_model):
    """
    Calculate model probability that actual temp meets threshold condition.
    
    Uses the error distribution from specified lead time.
    
    Args:
        forecast_temp: Forecasted temperature
        threshold: Threshold value (e.g., 44 for "44-45" or "≥44")
        threshold_type: 'above', 'below', or 'range'
        error_model: Loaded error model for specified lead time
    
    Returns:
        Probability (0 to 1)
    """
    mean_error = error_model['mean']  # Bias
    std_error = error_model['std']
    
    # Adjust forecast for bias
    adjusted_forecast = forecast_temp - mean_error
    
    if threshold_type == 'above':
        # P(actual >= threshold)
        z_score = (threshold - adjusted_forecast) / std_error
        return 1 - stats.norm.cdf(z_score)
    
    elif threshold_type == 'below':
        # P(actual <= threshold)
        z_score = (threshold - adjusted_forecast) / std_error
        return stats.norm.cdf(z_score)
    
    elif threshold_type == 'range':
        # For range like "44-45", need to parse it
        # Polymarket rounds to whole degrees, so 44-45 means 44.5 <= actual < 45.5
        low = threshold - 0.5
        high = threshold + 0.5
        
        z_low = (low - adjusted_forecast) / std_error
        z_high = (high - adjusted_forecast) / std_error
        
        return stats.norm.cdf(z_high) - stats.norm.cdf(z_low)
    
    return 0.0


def parse_threshold(threshold_str):
    """
    Parse threshold string to extract value and type.
    
    Examples:
        "≥44" -> (44, 'above')
        "≤35" -> (35, 'below')
        "44-45" -> (44.5, 'range')  # midpoint
    """
    threshold_str = str(threshold_str).strip()
    
    if '≥' in threshold_str or '>=' in threshold_str:
        value = float(threshold_str.replace('≥', '').replace('>=', '').strip())
        return value, 'above'
    
    elif '≤' in threshold_str or '<=' in threshold_str:
        value = float(threshold_str.replace('≤', '').replace('<=', '').strip())
        return value, 'below'
    
    elif '-' in threshold_str:
        # Range like "44-45"
        parts = threshold_str.split('-')
        low = float(parts[0].strip())
        high = float(parts[1].strip())
        midpoint = (low + high) / 2
        return midpoint, 'range'
    
    else:
        # Single value, assume exact
        return float(threshold_str), 'range'


def check_bet_outcome(actual_temp, threshold, threshold_type):
    """
    Check if bet would have won based on actual temperature.
    
    Args:
        actual_temp: Actual maximum temperature
        threshold: Threshold value
        threshold_type: 'above', 'below', or 'range'
    
    Returns:
        True if bet wins, False if bet loses
    """
    if threshold_type == 'above':
        return actual_temp >= threshold
    
    elif threshold_type == 'below':
        return actual_temp <= threshold
    
    elif threshold_type == 'range':
        # For range, check if actual rounds to the threshold
        # Polymarket uses whole degree rounding
        low = threshold - 0.5
        high = threshold + 0.5
        return low <= actual_temp < high
    
    return False


def backtest_strategy(lead_times=['1d', '0d'], min_edge=0.05, min_ev=0.05, bankroll=1000, kelly_fraction=0.25, max_bet_pct=0.05, min_market_prob=0.05, min_volume=100, max_bet_vs_volume=0.10):
    """
    Backtest the betting strategy on historical data using multiple lead times.
    
    Args:
        lead_times: List of lead times to use (e.g., ['1d', '0d'] for 1-day and same-day forecasts)
        min_edge: Minimum edge required (model_prob - market_prob)
        min_ev: Minimum expected value required
        bankroll: Starting bankroll
        kelly_fraction: Fraction of Kelly criterion to use (0.25 = quarter Kelly)
        max_bet_pct: Maximum bet as percentage of bankroll (default 5%)
        min_market_prob: Minimum market probability to consider (avoid illiquid markets)
        min_volume: Minimum market volume required (default $100)
        max_bet_vs_volume: Maximum bet size as fraction of market volume (default 10%)
    
    Returns:
        DataFrame with all betting opportunities and results
    """
    
    print("="*70)
    print("BACKTESTING BETTING STRATEGY")
    print("="*70)
    
    # Load error models for each lead time
    error_models = {}
    print(f"\nError Models:")
    for lead_time in lead_times:
        error_models[lead_time] = load_error_model(lead_time)
        model = error_models[lead_time]
        lead_name = "Same-day" if lead_time == '0d' else "1-day ahead"
        print(f"\n  {lead_name} ({lead_time}):")
        print(f"    Bias (Mean Error): {model['mean']:+.2f}°F")
        print(f"    Std Dev:           {model['std']:.2f}°F")
        print(f"    MAE:               {model['mae']:.2f}°F")
        print(f"    Sample Size:       {model['sample_size']} forecasts")
    
    # Load data
    print(f"\nLoading data...")
    
    # Load forecasts (handle both old and new formats)
    forecast_files = [
        'data/raw/historical_forecasts.csv',
        'data/raw/openmeteo_previous_runs.csv'
    ]
    
    forecasts = None
    for file_path in forecast_files:
        from pathlib import Path
        if Path(file_path).exists():
            forecasts = pd.read_csv(file_path)
            print(f"  Loaded forecasts from: {file_path}")
            break
    
    if forecasts is None:
        raise FileNotFoundError("No forecast data found")
    
    # Parse timestamps and normalize column names
    forecasts['valid_time'] = pd.to_datetime(forecasts['valid_time'])
    
    # Handle different CSV formats
    if 'lead_time' in forecasts.columns:
        forecasts['forecast_issued'] = pd.to_datetime(forecasts['forecast_issued'])
    elif 'days_before' in forecasts.columns:
        forecasts['lead_time'] = forecasts['days_before']
        forecasts['forecast_issued'] = pd.to_datetime(
            forecasts['forecast_date'] + ' ' + forecasts['forecast_time']
        )
        forecasts['forecast_datetime'] = forecasts['forecast_issued']
    
    # Filter to specified lead times
    lead_time_nums = [int(lt[0]) for lt in lead_times]  # Convert '1d' -> 1, '0d' -> 0
    forecasts_filtered = forecasts[forecasts['lead_time'].isin(lead_time_nums)].copy()
    print(f"  Using lead times {lead_times}: {len(forecasts_filtered):,} records")
    for lt_num in lead_time_nums:
        count = len(forecasts_filtered[forecasts_filtered['lead_time'] == lt_num])
        print(f"    {lt_num}-day: {count:,} records")
    
    # Load actual daily max temperatures from Weather Underground (official Polymarket source)
    try:
        daily_max = pd.read_csv('data/raw/wunderground_daily_max_temps.csv')
        daily_max['date'] = pd.to_datetime(daily_max['date'])
        print("Using Weather Underground daily max data (official Polymarket source)")
        use_wunderground = True
    except FileNotFoundError:
        print("Weather Underground data not found, falling back to Open-Meteo hourly data")
        actuals = pd.read_csv('data/raw/actual_temperatures.csv')
        actuals['timestamp'] = pd.to_datetime(actuals['timestamp'])
        use_wunderground = False
    
    odds_df = pd.read_csv('data/raw/polymarket_odds_history.csv')
    odds_df['event_date'] = pd.to_datetime(odds_df['event_date'])
    
    print(f"  Daily Max Temps: {len(daily_max):,} days")
    print(f"  Odds: {len(odds_df):,} records")
    
    # Get unique betting days (days with odds data)
    betting_days = odds_df['event_date'].unique()
    print(f"\nBetting days with odds: {len(betting_days)}")
    
    # Strategy parameters
    print(f"\nStrategy Parameters:")
    print(f"  Min Edge: {min_edge:.1%}")
    print(f"  Min EV: {min_ev:.1%}")
    print(f"  Min Market Prob: {min_market_prob:.1%}")
    print(f"  Min Volume: ${min_volume:,.0f}")
    print(f"  Max Bet vs Volume: {max_bet_vs_volume:.1%}")
    print(f"  Bankroll: ${bankroll:,.0f}")
    print(f"  Kelly Fraction: {kelly_fraction:.1%}")
    print(f"  Max Bet %: {max_bet_pct:.1%}")
    
    # Track all opportunities
    opportunities = []
    current_bankroll = bankroll
    
    print(f"\n{'='*70}")
    print("ANALYZING BETTING OPPORTUNITIES...")
    print(f"{'='*70}\n")
    
    for betting_day in sorted(betting_days):
        betting_day_dt = pd.to_datetime(betting_day)
        
        # Get forecasts for this betting day at different lead times
        betting_day_forecasts = forecasts_filtered[
            forecasts_filtered['valid_time'].dt.date == betting_day_dt.date()
        ]
        
        if len(betting_day_forecasts) == 0:
            continue
        
        # Group by lead time to get best forecast for each
        for lead_time_num in lead_time_nums:
            lead_time_str = f"{lead_time_num}d"
            
            # Get forecasts at this lead time
            lt_forecasts = betting_day_forecasts[
                betting_day_forecasts['lead_time'] == lead_time_num
            ]
            
            if len(lt_forecasts) == 0:
                continue
            
            forecasted_max = lt_forecasts['temperature'].max()
            forecast_issued = lt_forecasts['forecast_issued'].iloc[0]
            error_model = error_models[lead_time_str]
        
            # Get actual max for betting day from Weather Underground
            if use_wunderground:
                day_max = daily_max[daily_max['date'].dt.date == betting_day_dt.date()]
                if len(day_max) == 0:
                    continue
                actual_max = day_max['max_temp_f'].iloc[0]
            else:
                # Fallback to calculating from hourly data
                betting_day_actuals = actuals[
                    actuals['timestamp'].dt.date == betting_day_dt.date()
                ]
                if len(betting_day_actuals) == 0:
                    continue
                actual_max = betting_day_actuals['temperature_f'].max()
            
            # Get odds at the time when forecast was issued
            # We want odds closest to when the forecast was issued
            target_time = pd.to_datetime(forecast_issued)
            if target_time.tzinfo is None:
                target_time = target_time.tz_localize('America/New_York')
            
            day_odds = odds_df[odds_df['event_date'] == betting_day].copy()
            
            if len(day_odds) == 0:
                continue
            
            # Parse fetch_timestamp with UTC and convert to Eastern time
            day_odds['fetch_time'] = pd.to_datetime(day_odds['fetch_timestamp'], utc=True).dt.tz_convert('America/New_York')
            day_odds['time_diff'] = abs((day_odds['fetch_time'] - target_time).dt.total_seconds())
            
            # Get the odds snapshot closest to forecast time for each threshold
            # Group by threshold and take the row with minimum time difference
            day_odds_snapshot = day_odds.sort_values('time_diff').groupby('threshold').first().reset_index()
            
            # Analyze each threshold
            for _, odds_row in day_odds_snapshot.iterrows():
                threshold_str = odds_row['threshold']
                market_prob = odds_row['yes_probability']
                market_volume = odds_row.get('volume', 0)
                
                # Parse threshold
                try:
                    threshold_value, threshold_type = parse_threshold(threshold_str)
                except:
                    continue
                
                # Calculate model probability
                model_prob = calculate_model_probability(
                    forecasted_max, threshold_value, threshold_type, error_model
                )
                
                # Calculate edge and EV
                edge = model_prob - market_prob
                
                if market_prob > 0 and market_prob < 1:
                    payout_multiplier = 1 / market_prob
                    ev = (model_prob * payout_multiplier) - 1
                else:
                    ev = 0
                
                # Check if bet meets criteria (including liquidity)
                should_bet = (
                    (edge >= min_edge) and 
                    (ev >= min_ev) and 
                    (market_prob >= min_market_prob) and
                    (market_prob <= 0.95) and  # Avoid near-certain markets
                    (market_volume >= min_volume)  # Require minimum liquidity
                )
                
                # Calculate bet size using Kelly criterion
                bet_size = 0
                liquidity_constrained = False
                if should_bet and market_prob > 0 and market_prob < 1:
                    # Kelly formula: f = (bp - q) / b
                    # where b = odds-1, p = model_prob, q = 1-model_prob
                    b = (1 / market_prob) - 1
                    kelly = (b * model_prob - (1 - model_prob)) / b
                    kelly = max(0, min(kelly, 1))  # Clamp between 0 and 1
                    bet_size = kelly * kelly_fraction * current_bankroll
                    bet_size = min(bet_size, current_bankroll * max_bet_pct)  # Cap at max_bet_pct of bankroll
                    
                    # Apply liquidity constraint - can't bet more than X% of market volume
                    max_bet_from_volume = market_volume * max_bet_vs_volume
                    if bet_size > max_bet_from_volume:
                        bet_size = max_bet_from_volume
                        liquidity_constrained = True
                
                # Check outcome
                bet_won = check_bet_outcome(actual_max, threshold_value, threshold_type)
                
                # Calculate profit/loss
                if should_bet and bet_size > 0:
                    if bet_won:
                        profit = bet_size * ((1 / market_prob) - 1)
                    else:
                        profit = -bet_size
                    
                    current_bankroll += profit
                else:
                    profit = 0
                
                # Record opportunity
                opportunities.append({
                    'date': betting_day,
                    'lead_time': lead_time_str,
                    'forecast_issued': forecast_issued,
                    'odds_fetch_time': odds_row['fetch_time'],
                    'forecasted_max': forecasted_max,
                    'actual_max': actual_max,
                    'threshold': threshold_str,
                    'threshold_value': threshold_value,
                    'threshold_type': threshold_type,
                    'market_prob': market_prob,
                    'market_volume': market_volume,
                    'model_prob': model_prob,
                    'edge': edge,
                    'ev': ev,
                    'should_bet': should_bet,
                    'bet_size': bet_size,
                    'liquidity_constrained': liquidity_constrained,
                    'bet_won': bet_won,
                    'profit': profit,
                    'bankroll': current_bankroll
                })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(opportunities)
    
    if len(results_df) == 0:
        print("No betting opportunities found!")
        return results_df
    
    # Print summary
    print(f"\n{'='*70}")
    print("BACKTEST RESULTS")
    print(f"{'='*70}")
    
    total_opportunities = len(results_df)
    bets_placed = results_df['should_bet'].sum()
    bets_won = results_df[results_df['should_bet']]['bet_won'].sum()
    liquidity_constrained_count = results_df[results_df['should_bet']]['liquidity_constrained'].sum()
    
    print(f"\nTotal Opportunities Analyzed: {total_opportunities}")
    print(f"Bets Placed: {bets_placed}")
    print(f"Bets Won: {bets_won}")
    print(f"Win Rate: {(bets_won / bets_placed * 100) if bets_placed > 0 else 0:.1f}%")
    print(f"Liquidity Constrained Bets: {liquidity_constrained_count} ({(liquidity_constrained_count / bets_placed * 100) if bets_placed > 0 else 0:.1f}%)")
    
    total_profit = results_df[results_df['should_bet']]['profit'].sum()
    roi = (total_profit / bankroll) * 100
    
    print(f"\nFinancial Results:")
    print(f"  Starting Bankroll: ${bankroll:,.2f}")
    print(f"  Ending Bankroll:   ${current_bankroll:,.2f}")
    print(f"  Total Profit:      ${total_profit:+,.2f}")
    print(f"  ROI:               {roi:+.1f}%")
    
    # Save results
    output_path = 'data/results/backtest_results.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\nDetailed results saved to: {output_path}")
    
    # Show sample bets
    if bets_placed > 0:
        print(f"\n{'='*70}")
        print("SAMPLE BETS (First 10)")
        print(f"{'='*70}\n")
        
        sample_bets = results_df[results_df['should_bet']].head(10)
        for _, bet in sample_bets.iterrows():
            result = "✅ WON" if bet['bet_won'] else "❌ LOST"
            liquidity_note = " [LIQUIDITY CAPPED]" if bet['liquidity_constrained'] else ""
            print(f"{bet['date'].strftime('%Y-%m-%d')} ({bet['lead_time']}): {bet['threshold']}")
            print(f"  Forecast: {bet['forecasted_max']:.1f}°F, Actual: {bet['actual_max']:.1f}°F")
            print(f"  Model: {bet['model_prob']:.1%}, Market: {bet['market_prob']:.1%}, Edge: {bet['edge']:+.1%}")
            print(f"  Volume: ${bet['market_volume']:.0f}, Bet: ${bet['bet_size']:.2f}{liquidity_note}, {result}, Profit: ${bet['profit']:+.2f}")
            print()
    
    return results_df


if __name__ == '__main__':
    # Run backtest with default parameters
    results = backtest_strategy(
        lead_times=['1d', '0d'],  # Use 1-day and same-day forecasts
        min_edge=0.05,            # Require 5% edge
        min_ev=0.05,              # Require 5% expected value
        min_market_prob=0.05,     # Avoid illiquid markets below 5%
        min_volume=100,           # Require at least $100 volume
        max_bet_vs_volume=0.10,   # Max bet = 10% of market volume
        bankroll=1000,            # Start with $1000
        kelly_fraction=0.25,      # Use quarter Kelly
        max_bet_pct=0.05          # Max 5% of bankroll per bet
    )
    
    print(f"\n{'='*70}")
    print("Backtest complete! Check data/results/backtest_results.csv for details")
    print(f"{'='*70}")

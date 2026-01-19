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


def load_error_model():
    """Load the saved error distribution model"""
    with open('data/processed/error_distribution_analysis.json', 'r') as f:
        return json.load(f)


def calculate_model_probability(forecast_temp, threshold, threshold_type, error_model):
    """
    Calculate model probability that actual temp meets threshold condition.
    
    Args:
        forecast_temp: Forecasted temperature
        threshold: Threshold value (e.g., 44 for "44-45" or "≥44")
        threshold_type: 'above', 'below', or 'range'
        error_model: Loaded error model
    
    Returns:
        Probability (0 to 1)
    """
    mean_error = error_model['basic_statistics']['mean']
    std_error = error_model['basic_statistics']['std']
    
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


def backtest_strategy(min_edge=0.05, min_ev=0.05, bankroll=1000, kelly_fraction=0.25, max_bet_pct=0.05, min_market_prob=0.05):
    """
    Backtest the betting strategy on historical data.
    
    Args:
        min_edge: Minimum edge required (model_prob - market_prob)
        min_ev: Minimum expected value required
        bankroll: Starting bankroll
        kelly_fraction: Fraction of Kelly criterion to use (0.25 = quarter Kelly)
        max_bet_pct: Maximum bet as percentage of bankroll (default 5%)
        min_market_prob: Minimum market probability to consider (avoid illiquid markets)
    
    Returns:
        DataFrame with all betting opportunities and results
    """
    
    print("="*70)
    print("BACKTESTING BETTING STRATEGY")
    print("="*70)
    
    # Load error model
    error_model = load_error_model()
    print(f"\nError Model:")
    print(f"  Bias: {error_model['basic_statistics']['mean']:+.2f}°F")
    print(f"  Std:  {error_model['basic_statistics']['std']:.2f}°F")
    
    # Load data
    print(f"\nLoading data...")
    forecasts = pd.read_csv('data/raw/historical_forecasts.csv')
    forecasts['forecast_datetime'] = pd.to_datetime(
        forecasts['forecast_date'] + ' ' + forecasts['forecast_time']
    )
    forecasts['valid_time'] = pd.to_datetime(forecasts['valid_time'])
    
    actuals = pd.read_csv('data/raw/actual_temperatures.csv')
    actuals['timestamp'] = pd.to_datetime(actuals['timestamp'])
    
    odds_df = pd.read_csv('data/raw/polymarket_odds_history.csv')
    odds_df['event_date'] = pd.to_datetime(odds_df['event_date'])
    
    print(f"  Forecasts: {len(forecasts):,} records")
    print(f"  Actuals: {len(actuals):,} records")
    print(f"  Odds: {len(odds_df):,} records")
    
    # Get unique betting days (days with odds data)
    betting_days = odds_df['event_date'].unique()
    print(f"\nBetting days with odds: {len(betting_days)}")
    
    # Strategy parameters
    print(f"\nStrategy Parameters:")
    print(f"  Min Edge: {min_edge:.1%}")
    print(f"  Min EV: {min_ev:.1%}")
    print(f"  Min Market Prob: {min_market_prob:.1%}")
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
        
        # Get 9 PM forecast from day before
        forecast_day = betting_day_dt - timedelta(days=1)
        forecast_day_str = forecast_day.strftime('%Y-%m-%d')
        
        # Get 9 PM forecast
        evening_forecast = forecasts[
            (forecasts['forecast_date'] == forecast_day_str) &
            (forecasts['forecast_time'] == '21:00')
        ].copy()
        
        if len(evening_forecast) == 0:
            continue
        
        # Get forecasted max for betting day
        betting_day_forecasts = evening_forecast[
            evening_forecast['valid_time'].dt.date == betting_day_dt.date()
        ]
        
        if len(betting_day_forecasts) == 0:
            continue
        
        forecasted_max = betting_day_forecasts['temperature'].max()
        
        # Get actual max for betting day
        betting_day_actuals = actuals[
            actuals['timestamp'].dt.date == betting_day_dt.date()
        ]
        
        if len(betting_day_actuals) == 0:
            continue
        
        actual_max = betting_day_actuals['temperature_f'].max()
        
        # Get odds at 9 PM the day before (when we'd actually be betting)
        # We want odds closest to 9 PM on forecast_day
        target_time = pd.to_datetime(f"{forecast_day_str} 21:00:00").tz_localize('America/New_York')
        
        day_odds = odds_df[odds_df['event_date'] == betting_day].copy()
        
        if len(day_odds) == 0:
            continue
        
        # Parse fetch_timestamp with UTC and convert to Eastern time
        day_odds['fetch_time'] = pd.to_datetime(day_odds['fetch_timestamp'], utc=True).dt.tz_convert('America/New_York')
        day_odds['time_diff'] = abs((day_odds['fetch_time'] - target_time).dt.total_seconds())
        
        # Get the odds snapshot closest to 9 PM for each threshold
        # Group by threshold and take the row with minimum time difference
        day_odds = day_odds.sort_values('time_diff').groupby('threshold').first().reset_index()
        
        # Analyze each threshold
        for _, odds_row in day_odds.iterrows():
            threshold_str = odds_row['threshold']
            market_prob = odds_row['yes_probability']
            
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
            
            # Check if bet meets criteria
            should_bet = (
                (edge >= min_edge) and 
                (ev >= min_ev) and 
                (market_prob >= min_market_prob) and
                (market_prob <= 0.95)  # Avoid near-certain markets
            )
            
            # Calculate bet size using Kelly criterion
            bet_size = 0
            if should_bet and market_prob > 0 and market_prob < 1:
                # Kelly formula: f = (bp - q) / b
                # where b = odds-1, p = model_prob, q = 1-model_prob
                b = (1 / market_prob) - 1
                kelly = (b * model_prob - (1 - model_prob)) / b
                kelly = max(0, min(kelly, 1))  # Clamp between 0 and 1
                bet_size = kelly * kelly_fraction * current_bankroll
                bet_size = min(bet_size, current_bankroll * max_bet_pct)  # Cap at max_bet_pct of bankroll
            
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
                'forecast_date': forecast_day_str,
                'forecast_time': '21:00',
                'odds_fetch_time': odds_row['fetch_time'],
                'forecasted_max': forecasted_max,
                'actual_max': actual_max,
                'threshold': threshold_str,
                'threshold_value': threshold_value,
                'threshold_type': threshold_type,
                'market_prob': market_prob,
                'model_prob': model_prob,
                'edge': edge,
                'ev': ev,
                'should_bet': should_bet,
                'bet_size': bet_size,
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
    
    print(f"\nTotal Opportunities Analyzed: {total_opportunities}")
    print(f"Bets Placed: {bets_placed}")
    print(f"Bets Won: {bets_won}")
    print(f"Win Rate: {(bets_won / bets_placed * 100) if bets_placed > 0 else 0:.1f}%")
    
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
            print(f"{bet['date'].strftime('%Y-%m-%d')}: {bet['threshold']}")
            print(f"  Forecast: {bet['forecasted_max']:.1f}°F, Actual: {bet['actual_max']:.1f}°F")
            print(f"  Model: {bet['model_prob']:.1%}, Market: {bet['market_prob']:.1%}, Edge: {bet['edge']:+.1%}")
            print(f"  Bet: ${bet['bet_size']:.2f}, {result}, Profit: ${bet['profit']:+.2f}")
            print()
    
    return results_df


if __name__ == '__main__':
    # Run backtest with default parameters
    results = backtest_strategy(
        min_edge=0.05,          # Require 5% edge
        min_ev=0.05,            # Require 5% expected value
        min_market_prob=0.05,   # Avoid illiquid markets below 5%
        bankroll=1000,          # Start with $1000
        kelly_fraction=0.25,    # Use quarter Kelly
        max_bet_pct=0.05        # Max 5% of bankroll per bet
    )
    
    print(f"\n{'='*70}")
    print("Backtest complete! Check data/results/backtest_results.csv for details")
    print(f"{'='*70}")

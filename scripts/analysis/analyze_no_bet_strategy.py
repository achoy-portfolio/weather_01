"""
Analyze NO Bet Strategy Performance

This script analyzes the performance of NO bets vs YES bets
to understand when betting NO on temperature ranges is profitable.
"""

import pandas as pd
import numpy as np

def analyze_no_bet_performance():
    """Analyze NO bet strategy from backtest results."""
    
    # Load backtest results
    results = pd.read_csv('data/results/backtest_results.csv')
    
    # Filter to only bets that were placed
    bets = results[results['should_bet'] == True].copy()
    
    print("="*70)
    print("NO BET STRATEGY ANALYSIS")
    print("="*70)
    
    # Overall statistics
    yes_bets = bets[bets['bet_side'] == 'YES']
    no_bets = bets[bets['bet_side'] == 'NO']
    
    print(f"\n{'='*70}")
    print("OVERALL PERFORMANCE")
    print(f"{'='*70}")
    
    print(f"\nYES Bets:")
    print(f"  Count: {len(yes_bets)}")
    print(f"  Win Rate: {yes_bets['bet_won'].mean():.1%}")
    print(f"  Avg Edge: {yes_bets['edge'].mean():.1%}")
    print(f"  Avg EV: {yes_bets['ev'].mean():.1%}")
    print(f"  Total Profit: ${yes_bets['profit'].sum():,.2f}")
    print(f"  Avg Profit per Bet: ${yes_bets['profit'].mean():,.2f}")
    
    print(f"\nNO Bets:")
    print(f"  Count: {len(no_bets)}")
    print(f"  Win Rate: {no_bets['bet_won'].mean():.1%}")
    print(f"  Avg Edge: {no_bets['edge'].mean():.1%}")
    print(f"  Avg EV: {no_bets['ev'].mean():.1%}")
    print(f"  Total Profit: ${no_bets['profit'].sum():,.2f}")
    print(f"  Avg Profit per Bet: ${no_bets['profit'].mean():,.2f}")
    
    # Analyze by distance from forecast
    print(f"\n{'='*70}")
    print("NO BETS BY DISTANCE FROM FORECAST")
    print(f"{'='*70}")
    
    no_bets['distance'] = no_bets['forecasted_max'] - no_bets['threshold_value']
    
    # Group by distance buckets
    distance_bins = [2, 3, 4, 5, 10, 100]
    no_bets['distance_bucket'] = pd.cut(no_bets['distance'], 
                                         bins=distance_bins, 
                                         labels=['2-3°F', '3-4°F', '4-5°F', '5-10°F', '10+°F'])
    
    for bucket in ['2-3°F', '3-4°F', '4-5°F', '5-10°F', '10+°F']:
        bucket_bets = no_bets[no_bets['distance_bucket'] == bucket]
        if len(bucket_bets) > 0:
            print(f"\n{bucket} below forecast:")
            print(f"  Count: {len(bucket_bets)}")
            print(f"  Win Rate: {bucket_bets['bet_won'].mean():.1%}")
            print(f"  Avg Edge: {bucket_bets['edge'].mean():.1%}")
            print(f"  Total Profit: ${bucket_bets['profit'].sum():,.2f}")
    
    # Analyze by lead time
    print(f"\n{'='*70}")
    print("PERFORMANCE BY LEAD TIME")
    print(f"{'='*70}")
    
    for lead_time in ['1d', '0d']:
        lt_yes = yes_bets[yes_bets['lead_time'] == lead_time]
        lt_no = no_bets[no_bets['lead_time'] == lead_time]
        
        print(f"\n{lead_time} Lead Time:")
        print(f"  YES Bets: {len(lt_yes)}, Win Rate: {lt_yes['bet_won'].mean():.1%}, Profit: ${lt_yes['profit'].sum():,.2f}")
        print(f"  NO Bets: {len(lt_no)}, Win Rate: {lt_no['bet_won'].mean():.1%}, Profit: ${lt_no['profit'].sum():,.2f}")
    
    # Show examples of NO bets that lost
    print(f"\n{'='*70}")
    print("EXAMPLES OF NO BETS THAT LOST (Learning Opportunities)")
    print(f"{'='*70}")
    
    no_losses = no_bets[no_bets['bet_won'] == False].head(10)
    
    if len(no_losses) > 0:
        for _, bet in no_losses.iterrows():
            print(f"\n{bet['date']} ({bet['lead_time']}): NO on {bet['threshold']}")
            print(f"  Forecast: {bet['forecasted_max']:.1f}°F, Actual: {bet['actual_max']:.1f}°F")
            print(f"  Distance: {bet['distance']:.1f}°F below forecast")
            print(f"  Model Prob (NO wins): {bet['model_prob']:.1%}, Market: {bet['market_prob']:.1%}")
            print(f"  Loss: ${bet['profit']:.2f}")
    else:
        print("\nNo NO bets lost! Perfect record.")
    
    # Key insights
    print(f"\n{'='*70}")
    print("KEY INSIGHTS")
    print(f"{'='*70}")
    
    print("\n1. NO Bet Strategy:")
    print(f"   - Betting NO on ranges {no_bets['distance'].min():.1f}°F to {no_bets['distance'].max():.1f}°F below forecast")
    print(f"   - Win rate: {no_bets['bet_won'].mean():.1%}")
    print(f"   - Average edge: {no_bets['edge'].mean():.1%}")
    
    print("\n2. Comparison to YES Bets:")
    yes_roi = (yes_bets['profit'].sum() / (yes_bets['bet_size'].sum() - yes_bets['profit'].sum())) * 100 if len(yes_bets) > 0 else 0
    no_roi = (no_bets['profit'].sum() / (no_bets['bet_size'].sum() - no_bets['profit'].sum())) * 100 if len(no_bets) > 0 else 0
    print(f"   - YES bets ROI: {yes_roi:.1f}%")
    print(f"   - NO bets ROI: {no_roi:.1f}%")
    
    print("\n3. Optimal Strategy:")
    best_distance = no_bets.groupby('distance_bucket')['profit'].sum().idxmax()
    print(f"   - Most profitable distance: {best_distance}")
    print(f"   - Consider betting NO on ranges 2+ degrees below forecast")
    print(f"   - Higher confidence with same-day forecasts (lower error)")


if __name__ == '__main__':
    analyze_no_bet_performance()

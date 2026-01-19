"""
Betting Strategy Simulator

Analyze backtest results and simulate different strategy parameters.
"""

import pandas as pd
import numpy as np


def analyze_backtest_results(results_path='data/results/backtest_results.csv'):
    """Analyze the backtest results in detail"""
    
    df = pd.read_csv(results_path)
    df['date'] = pd.to_datetime(df['date'])
    
    print("="*70)
    print("BACKTEST RESULTS ANALYSIS")
    print("="*70)
    
    # Overall stats
    total_opps = len(df)
    bets_placed = df['should_bet'].sum()
    bets_won = df[df['should_bet']]['bet_won'].sum()
    
    print(f"\nOverall Statistics:")
    print(f"  Total Opportunities: {total_opps:,}")
    print(f"  Bets Placed: {bets_placed}")
    print(f"  Bets Won: {bets_won}")
    print(f"  Win Rate: {(bets_won / bets_placed * 100) if bets_placed > 0 else 0:.1f}%")
    
    # Financial results
    bets_df = df[df['should_bet']].copy()
    if len(bets_df) > 0:
        total_wagered = bets_df['bet_size'].sum()
        total_profit = bets_df['profit'].sum()
        roi = (total_profit / total_wagered) * 100 if total_wagered > 0 else 0
        
        print(f"\nFinancial Results:")
        print(f"  Total Wagered: ${total_wagered:,.2f}")
        print(f"  Total Profit: ${total_profit:+,.2f}")
        print(f"  ROI on Wagered: {roi:+.1f}%")
        print(f"  Avg Bet Size: ${bets_df['bet_size'].mean():.2f}")
        print(f"  Avg Profit per Bet: ${bets_df['profit'].mean():+.2f}")
    
    # Edge analysis
    print(f"\n{'='*70}")
    print("EDGE ANALYSIS")
    print(f"{'='*70}")
    
    if len(bets_df) > 0:
        print(f"\nEdge Statistics (for bets placed):")
        print(f"  Mean Edge: {bets_df['edge'].mean():.1%}")
        print(f"  Median Edge: {bets_df['edge'].median():.1%}")
        print(f"  Min Edge: {bets_df['edge'].min():.1%}")
        print(f"  Max Edge: {bets_df['edge'].max():.1%}")
        
        # Win rate by edge bucket
        print(f"\nWin Rate by Edge Bucket:")
        bets_df['edge_bucket'] = pd.cut(
            bets_df['edge'], 
            bins=[0, 0.1, 0.2, 0.3, 1.0],
            labels=['5-10%', '10-20%', '20-30%', '>30%']
        )
        
        for bucket in ['5-10%', '10-20%', '20-30%', '>30%']:
            bucket_bets = bets_df[bets_df['edge_bucket'] == bucket]
            if len(bucket_bets) > 0:
                win_rate = bucket_bets['bet_won'].sum() / len(bucket_bets) * 100
                avg_profit = bucket_bets['profit'].mean()
                print(f"  {bucket}: {len(bucket_bets)} bets, {win_rate:.1f}% win rate, ${avg_profit:+.2f} avg profit")
    
    # Threshold type analysis
    print(f"\n{'='*70}")
    print("THRESHOLD TYPE ANALYSIS")
    print(f"{'='*70}")
    
    if len(bets_df) > 0:
        for threshold_type in ['above', 'below', 'range']:
            type_bets = bets_df[bets_df['threshold_type'] == threshold_type]
            if len(type_bets) > 0:
                win_rate = type_bets['bet_won'].sum() / len(type_bets) * 100
                total_profit = type_bets['profit'].sum()
                print(f"\n{threshold_type.upper()}:")
                print(f"  Bets: {len(type_bets)}")
                print(f"  Win Rate: {win_rate:.1f}%")
                print(f"  Total Profit: ${total_profit:+,.2f}")
    
    # Monthly performance
    print(f"\n{'='*70}")
    print("MONTHLY PERFORMANCE")
    print(f"{'='*70}\n")
    
    if len(bets_df) > 0:
        bets_df['month'] = bets_df['date'].dt.to_period('M')
        monthly = bets_df.groupby('month').agg({
            'bet_size': 'count',
            'bet_won': 'sum',
            'profit': 'sum'
        }).rename(columns={'bet_size': 'bets'})
        
        monthly['win_rate'] = (monthly['bet_won'] / monthly['bets'] * 100).round(1)
        
        for month, row in monthly.iterrows():
            print(f"{month}: {int(row['bets'])} bets, {row['win_rate']:.1f}% win rate, ${row['profit']:+,.2f} profit")
    
    # Best and worst bets
    print(f"\n{'='*70}")
    print("BEST AND WORST BETS")
    print(f"{'='*70}")
    
    if len(bets_df) > 0:
        print(f"\nTop 5 Winning Bets:")
        top_wins = bets_df.nlargest(5, 'profit')
        for _, bet in top_wins.iterrows():
            print(f"  {bet['date'].strftime('%Y-%m-%d')}: {bet['threshold']}")
            print(f"    Forecast: {bet['forecasted_max']:.1f}°F, Actual: {bet['actual_max']:.1f}°F")
            print(f"    Edge: {bet['edge']:.1%}, Bet: ${bet['bet_size']:.2f}, Profit: ${bet['profit']:+,.2f}")
        
        print(f"\nTop 5 Losing Bets:")
        top_losses = bets_df.nsmallest(5, 'profit')
        for _, bet in top_losses.iterrows():
            print(f"  {bet['date'].strftime('%Y-%m-%d')}: {bet['threshold']}")
            print(f"    Forecast: {bet['forecasted_max']:.1f}°F, Actual: {bet['actual_max']:.1f}°F")
            print(f"    Edge: {bet['edge']:.1%}, Bet: ${bet['bet_size']:.2f}, Profit: ${bet['profit']:+,.2f}")


if __name__ == '__main__':
    analyze_backtest_results()

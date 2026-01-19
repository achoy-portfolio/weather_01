"""
Outcome Evaluation Script

This script evaluates betting outcomes by comparing betting decisions
against actual temperature observations and generates bet-by-bet results.
"""

import pandas as pd
import sys
import os
from datetime import datetime
from betting_simulator import BettingSimulator, OutcomeEvaluator


def load_betting_decisions(decisions_file: str) -> list:
    """
    Load betting decisions from a CSV or generate from scratch.
    
    Args:
        decisions_file: Path to betting decisions CSV
        
    Returns:
        List of betting decision dictionaries
    """
    if os.path.exists(decisions_file):
        df = pd.read_csv(decisions_file)
        return df.to_dict('records')
    else:
        print(f"Warning: {decisions_file} not found. Generate decisions first.")
        return []


def evaluate_and_save_results(
    betting_decisions: list,
    actual_temps_file: str,
    output_file: str,
    starting_bankroll: float = 1000.0
):
    """
    Evaluate betting outcomes and save results to CSV.
    
    Args:
        betting_decisions: List of betting decision dictionaries
        actual_temps_file: Path to actual temperatures CSV
        output_file: Path to save results CSV
        starting_bankroll: Starting bankroll in dollars
    """
    # Load actual temperatures
    print(f"Loading actual temperatures from {actual_temps_file}...")
    actual_temps_df = pd.read_csv(actual_temps_file)
    
    # Evaluate outcomes
    print(f"Evaluating {len(betting_decisions)} betting decisions...")
    evaluator = OutcomeEvaluator()
    results_df = evaluator.evaluate_betting_results(
        betting_decisions,
        actual_temps_df,
        starting_bankroll
    )
    
    # Save results
    print(f"Saving results to {output_file}...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    results_df.to_csv(output_file, index=False)
    
    # Generate summary
    summary = evaluator.generate_summary_statistics(results_df)
    
    # Display summary
    print("\n" + "=" * 70)
    print("BETTING RESULTS SUMMARY")
    print("=" * 70)
    print(f"Total Bets Placed: {summary['total_bets']}")
    print(f"Total Amount Wagered: ${summary['total_wagered']:.2f}")
    print(f"Total Profit/Loss: ${summary['total_profit_loss']:.2f}")
    print(f"Win Rate: {summary['win_rate']:.1%}")
    print(f"ROI: {summary['roi']:.1f}%")
    print(f"Final Cumulative P&L: ${summary['final_cumulative_pl']:.2f}")
    print("=" * 70)
    
    # Display bet-by-bet results for bets that were placed
    bets_placed = results_df[results_df['bet_placed'] == True]
    
    if not bets_placed.empty:
        print("\nBet-by-Bet Results:")
        print("-" * 70)
        
        for idx, row in bets_placed.iterrows():
            outcome_symbol = "✓" if row['bet_outcome'] == 'win' else "✗"
            print(f"{outcome_symbol} {row['target_date']} | {row['threshold']:>8} | "
                  f"Forecast: {row['forecast_temp']:>5.1f}°F | "
                  f"Actual: {row['actual_temp']:>5.1f}°F | "
                  f"Bet: ${row['bet_size']:>6.2f} | "
                  f"P&L: ${row['profit_loss']:>7.2f} | "
                  f"Cumulative: ${row['cumulative_pl']:>7.2f}")
    
    return results_df, summary


def main():
    """Main execution function"""
    print("=" * 70)
    print("Outcome Evaluator")
    print("=" * 70)
    print()
    
    # File paths
    actual_temps_file = 'data/raw/actual_temperatures.csv'
    decisions_file = 'data/results/betting_decisions.csv'
    output_file = 'data/results/betting_outcomes.csv'
    
    # Check if actual temperatures exist
    if not os.path.exists(actual_temps_file):
        print(f"Error: {actual_temps_file} not found.")
        print("Please run the actual temperature fetcher first.")
        sys.exit(1)
    
    # For demonstration, we'll simulate some betting decisions
    # In practice, these would come from the betting simulator
    print("Note: This is a demonstration. In production, load decisions from CSV.")
    print()
    
    # Load actual temperatures to get available dates
    actual_temps_df = pd.read_csv(actual_temps_file)
    actual_temps_df['timestamp'] = pd.to_datetime(actual_temps_df['timestamp'])
    available_dates = actual_temps_df['timestamp'].dt.date.unique()
    
    if len(available_dates) == 0:
        print("No actual temperature data available.")
        sys.exit(1)
    
    # Create sample betting decisions for demonstration
    sample_date = str(available_dates[0])
    
    # Get actual peak temp for the sample date
    date_temps = actual_temps_df[actual_temps_df['timestamp'].dt.date == available_dates[0]]
    peak_temp = date_temps['temperature_f'].max()
    
    print(f"Sample date: {sample_date}")
    print(f"Actual peak temperature: {peak_temp:.1f}°F")
    print()
    
    # Create sample decisions
    sample_decisions = [
        {
            'target_date': sample_date,
            'threshold': f'≥{int(peak_temp - 5)}',
            'threshold_type': 'above',
            'threshold_low': peak_temp - 5,
            'threshold_high': None,
            'forecast_temp': peak_temp - 2,
            'should_bet': True,
            'bet_size': 50.0,
            'model_probability': 0.7,
            'market_odds': 0.5,
            'expected_value': 0.4
        },
        {
            'target_date': sample_date,
            'threshold': f'≤{int(peak_temp - 10)}',
            'threshold_type': 'below',
            'threshold_low': peak_temp - 10,
            'threshold_high': None,
            'forecast_temp': peak_temp - 2,
            'should_bet': True,
            'bet_size': 30.0,
            'model_probability': 0.2,
            'market_odds': 0.4,
            'expected_value': -0.5
        }
    ]
    
    # Evaluate and save results
    results_df, summary = evaluate_and_save_results(
        sample_decisions,
        actual_temps_file,
        output_file,
        starting_bankroll=1000.0
    )
    
    print(f"\n✓ Results saved to {output_file}")
    print(f"✓ Generated {len(results_df)} result records")


if __name__ == "__main__":
    main()

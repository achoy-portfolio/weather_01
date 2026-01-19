"""
Betting Decision Simulator for Polymarket Temperature Markets

This module simulates betting decisions based on forecast data and market odds,
using statistical methods to calculate probabilities and expected values.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import re


class BettingSimulator:
    """
    Simulates betting decisions for Polymarket temperature markets using
    forecast data, actual temperatures, and market odds.
    """
    
    def __init__(
        self,
        ev_threshold: float = 0.05,
        kelly_fraction: float = 0.25,
        bankroll: float = 1000.0,
        base_uncertainty: float = 3.4
    ):
        """
        Initialize the betting simulator.
        
        Args:
            ev_threshold: Minimum expected value to place a bet (default 5%)
            kelly_fraction: Fraction of Kelly criterion to use (default 25%)
            bankroll: Starting bankroll in dollars
            base_uncertainty: Base forecast uncertainty in degrees F (default 3.4°F)
        """
        self.ev_threshold = ev_threshold
        self.kelly_fraction = kelly_fraction
        self.bankroll = bankroll
        self.base_uncertainty = base_uncertainty
    
    def parse_threshold(self, threshold_str: str) -> Tuple[str, float, Optional[float]]:
        """
        Parse threshold string to extract type and values.
        
        Args:
            threshold_str: Threshold string like "36-37", "≥75", "≤33"
            
        Returns:
            Tuple of (threshold_type, lower_bound, upper_bound)
            - For "above": ("above", 75.0, None)
            - For "below": ("below", 33.0, None)
            - For "range": ("range", 36.0, 37.0)
        """
        threshold_str = str(threshold_str).strip()
        
        # Check for range format (e.g., "36-37")
        range_match = re.match(r'^(\d+)-(\d+)$', threshold_str)
        if range_match:
            low = float(range_match.group(1))
            high = float(range_match.group(2))
            return ("range", low, high)
        
        # Check for "above" format (e.g., "≥75" or ">=75")
        above_match = re.match(r'^[≥>=]+(\d+)$', threshold_str)
        if above_match:
            value = float(above_match.group(1))
            return ("above", value, None)
        
        # Check for "below" format (e.g., "≤33" or "<=33")
        below_match = re.match(r'^[≤<=]+(\d+)$', threshold_str)
        if below_match:
            value = float(below_match.group(1))
            return ("below", value, None)
        
        # If no pattern matches, try to parse as a single number (treat as exact)
        try:
            value = float(threshold_str)
            return ("range", value, value)
        except ValueError:
            raise ValueError(f"Unable to parse threshold: {threshold_str}")
    
    def calculate_model_probability(
        self,
        forecast_temp: float,
        threshold_type: str,
        threshold_low: float,
        threshold_high: Optional[float] = None,
        uncertainty: Optional[float] = None
    ) -> float:
        """
        Calculate the probability that the actual temperature will satisfy
        the threshold condition, using a normal distribution assumption.
        
        Args:
            forecast_temp: Forecasted temperature in °F
            threshold_type: One of "above", "below", or "range"
            threshold_low: Lower bound of threshold
            threshold_high: Upper bound of threshold (for range type)
            uncertainty: Forecast uncertainty (std dev) in °F
            
        Returns:
            Probability between 0 and 1
        """
        if uncertainty is None:
            uncertainty = self.base_uncertainty
        
        if threshold_type == "above":
            # P(temp >= threshold)
            z_score = (threshold_low - forecast_temp) / uncertainty
            prob = 1 - stats.norm.cdf(z_score)
            
        elif threshold_type == "below":
            # P(temp <= threshold)
            z_score = (threshold_low - forecast_temp) / uncertainty
            prob = stats.norm.cdf(z_score)
            
        elif threshold_type == "range":
            # P(low <= temp <= high)
            # Account for Polymarket rounding: 36.5-37.4 rounds to 37°F
            # So for "36-37" bucket, we want P(35.5 <= temp < 37.5)
            adjusted_low = threshold_low - 0.5
            adjusted_high = threshold_high + 0.5 if threshold_high else threshold_low + 0.5
            
            z_low = (adjusted_low - forecast_temp) / uncertainty
            z_high = (adjusted_high - forecast_temp) / uncertainty
            
            prob = stats.norm.cdf(z_high) - stats.norm.cdf(z_low)
        else:
            raise ValueError(f"Unknown threshold type: {threshold_type}")
        
        # Clamp probability to valid range
        return max(0.0, min(1.0, prob))
    
    def calculate_expected_value(
        self,
        model_prob: float,
        market_odds: float
    ) -> float:
        """
        Calculate expected value of a bet.
        
        Args:
            model_prob: Model's estimated probability of winning (0-1)
            market_odds: Market's implied probability (0-1)
            
        Returns:
            Expected value as a decimal (e.g., 0.15 = 15% EV)
        """
        # Payout multiplier is 1 / market_odds
        # If market_odds = 0.5, you get 2x your money back (1/0.5 = 2)
        payout_multiplier = 1.0 / market_odds if market_odds > 0 else 0
        
        # EV = (model_prob × payout_multiplier) - 1
        # This represents the expected profit per dollar bet
        ev = (model_prob * payout_multiplier) - 1.0
        
        return ev
    
    def calculate_kelly_bet_size(
        self,
        model_prob: float,
        market_odds: float,
        bankroll: float
    ) -> float:
        """
        Calculate bet size using Kelly Criterion with fractional Kelly.
        
        Args:
            model_prob: Model's estimated probability of winning (0-1)
            market_odds: Market's implied probability (0-1)
            bankroll: Current bankroll in dollars
            
        Returns:
            Bet size in dollars
        """
        # Payout odds (b) = (1/market_odds) - 1
        # For market_odds = 0.5, b = 1 (even money)
        b = (1.0 / market_odds) - 1.0 if market_odds > 0 else 0
        
        # Kelly fraction: f = (b*p - q) / b
        # where p = model_prob, q = 1 - model_prob
        q = 1.0 - model_prob
        
        if b > 0:
            kelly_fraction = (b * model_prob - q) / b
        else:
            kelly_fraction = 0
        
        # Apply fractional Kelly for risk management
        kelly_fraction = kelly_fraction * self.kelly_fraction
        
        # Ensure non-negative bet size
        kelly_fraction = max(0, kelly_fraction)
        
        # Calculate bet size
        bet_size = kelly_fraction * bankroll
        
        return bet_size
    
    def should_place_bet(
        self,
        forecast_temp: float,
        threshold: str,
        market_odds: float,
        uncertainty: Optional[float] = None
    ) -> Dict:
        """
        Determine whether to place a bet based on expected value.
        
        Args:
            forecast_temp: Forecasted temperature in °F
            threshold: Threshold string (e.g., "36-37", "≥75")
            market_odds: Market's implied probability (0-1)
            uncertainty: Forecast uncertainty in °F
            
        Returns:
            Dictionary with betting decision details:
            {
                'should_bet': bool,
                'bet_size': float,
                'expected_value': float,
                'model_probability': float,
                'threshold_type': str,
                'threshold_low': float,
                'threshold_high': Optional[float]
            }
        """
        # Parse threshold
        threshold_type, threshold_low, threshold_high = self.parse_threshold(threshold)
        
        # Calculate model probability
        model_prob = self.calculate_model_probability(
            forecast_temp,
            threshold_type,
            threshold_low,
            threshold_high,
            uncertainty
        )
        
        # Calculate expected value
        ev = self.calculate_expected_value(model_prob, market_odds)
        
        # Decide whether to bet
        should_bet = ev > self.ev_threshold
        
        # Calculate bet size if betting
        bet_size = 0.0
        if should_bet:
            bet_size = self.calculate_kelly_bet_size(
                model_prob,
                market_odds,
                self.bankroll
            )
        
        return {
            'should_bet': should_bet,
            'bet_size': bet_size,
            'expected_value': ev,
            'model_probability': model_prob,
            'threshold_type': threshold_type,
            'threshold_low': threshold_low,
            'threshold_high': threshold_high,
            'market_odds': market_odds,
            'forecast_temp': forecast_temp
        }
    
    def simulate_day_betting_decisions(
        self,
        target_date: str,
        forecast_temp: float,
        market_odds_df: pd.DataFrame,
        uncertainty: Optional[float] = None
    ) -> List[Dict]:
        """
        Simulate betting decisions for all thresholds on a given day.
        
        Args:
            target_date: Target date in YYYY-MM-DD format
            forecast_temp: Forecasted peak temperature for the day
            market_odds_df: DataFrame with columns ['threshold', 'yes_probability']
            uncertainty: Forecast uncertainty in °F
            
        Returns:
            List of betting decisions, one per threshold
        """
        decisions = []
        
        for _, row in market_odds_df.iterrows():
            threshold = row['threshold']
            market_odds = row['yes_probability']
            
            # Skip if market odds are invalid
            if pd.isna(market_odds) or market_odds <= 0 or market_odds >= 1:
                continue
            
            decision = self.should_place_bet(
                forecast_temp,
                threshold,
                market_odds,
                uncertainty
            )
            
            # Add metadata
            decision['target_date'] = target_date
            decision['threshold'] = threshold
            
            decisions.append(decision)
        
        return decisions


def load_forecast_for_date(
    forecast_df: pd.DataFrame,
    target_date: str,
    forecast_time: str = "21:00"
) -> Optional[float]:
    """
    Load the forecasted peak temperature for a target date.
    
    Args:
        forecast_df: DataFrame with historical forecasts
        target_date: Target date in YYYY-MM-DD format
        forecast_time: Time of forecast (default "21:00" for 9 PM)
        
    Returns:
        Forecasted peak temperature, or None if not found
    """
    # Convert target_date to datetime for comparison
    target_dt = pd.to_datetime(target_date)
    
    # Filter for forecasts issued the day before at 9 PM
    forecast_date = (target_dt - timedelta(days=1)).strftime('%Y-%m-%d')
    
    # Filter forecast data
    mask = (
        (forecast_df['forecast_date'] == forecast_date) &
        (forecast_df['forecast_time'] == forecast_time)
    )
    
    forecast_data = forecast_df[mask].copy()
    
    if forecast_data.empty:
        return None
    
    # Get the maximum forecasted temperature for the target day
    # Filter for valid_time on the target date
    forecast_data['valid_time'] = pd.to_datetime(forecast_data['valid_time'])
    target_forecasts = forecast_data[
        forecast_data['valid_time'].dt.date == target_dt.date()
    ]
    
    if target_forecasts.empty:
        return None
    
    # Return the maximum temperature
    return target_forecasts['temperature'].max()


def load_market_odds_for_date(
    odds_df: pd.DataFrame,
    target_date: str,
    fetch_time: Optional[str] = None
) -> pd.DataFrame:
    """
    Load market odds for a target date.
    
    Args:
        odds_df: DataFrame with historical odds
        target_date: Target date in YYYY-MM-DD format
        fetch_time: Specific fetch time to use (optional)
        
    Returns:
        DataFrame with columns ['threshold', 'yes_probability', 'threshold_type']
    """
    # Filter for target date
    date_odds = odds_df[odds_df['event_date'] == target_date].copy()
    
    if date_odds.empty:
        return pd.DataFrame(columns=['threshold', 'yes_probability', 'threshold_type'])
    
    # If fetch_time specified, filter for that time
    if fetch_time:
        date_odds = date_odds[date_odds['fetch_timestamp'] == fetch_time]
    else:
        # Use the earliest available odds (market open)
        date_odds = date_odds.sort_values('fetch_timestamp')
        earliest_time = date_odds['fetch_timestamp'].iloc[0]
        date_odds = date_odds[date_odds['fetch_timestamp'] == earliest_time]
    
    # Get unique thresholds with their odds
    result = date_odds.groupby(['threshold', 'threshold_type']).agg({
        'yes_probability': 'first'
    }).reset_index()
    
    return result


class OutcomeEvaluator:
    """
    Evaluates betting outcomes based on actual temperatures and calculates P&L.
    """
    
    @staticmethod
    def apply_polymarket_rounding(temp: float) -> int:
        """
        Apply Polymarket rounding rules (whole degree resolution).
        
        Polymarket rounds to the nearest whole degree using standard rounding:
        - 34.5 rounds to 35
        - 34.4 rounds to 34
        
        Args:
            temp: Temperature in degrees F
            
        Returns:
            Rounded temperature as integer
        """
        return round(temp)
    
    @staticmethod
    def evaluate_bet_outcome(
        actual_temp: float,
        threshold_type: str,
        threshold_low: float,
        threshold_high: Optional[float] = None
    ) -> bool:
        """
        Determine if a bet would have won based on actual temperature.
        
        Args:
            actual_temp: Actual observed temperature in °F
            threshold_type: One of "above", "below", or "range"
            threshold_low: Lower bound of threshold
            threshold_high: Upper bound of threshold (for range type)
            
        Returns:
            True if bet won, False if bet lost
        """
        # Apply Polymarket rounding to actual temperature
        rounded_temp = OutcomeEvaluator.apply_polymarket_rounding(actual_temp)
        
        if threshold_type == "above":
            # For "≥75", bet wins if rounded temp >= 75
            return rounded_temp >= threshold_low
            
        elif threshold_type == "below":
            # For "≤33", bet wins if rounded temp <= 33
            return rounded_temp <= threshold_low
            
        elif threshold_type == "range":
            # For "36-37", bet wins if rounded temp is 36 or 37
            # This accounts for the fact that 35.5-37.4 rounds to 36 or 37
            if threshold_high is None:
                # Single value range
                return rounded_temp == threshold_low
            else:
                return threshold_low <= rounded_temp <= threshold_high
        else:
            raise ValueError(f"Unknown threshold type: {threshold_type}")
    
    @staticmethod
    def calculate_profit_loss(
        bet_size: float,
        bet_won: bool,
        market_odds: float
    ) -> float:
        """
        Calculate profit or loss for a bet.
        
        Args:
            bet_size: Amount bet in dollars
            bet_won: Whether the bet won
            market_odds: Market's implied probability (0-1)
            
        Returns:
            Profit (positive) or loss (negative) in dollars
        """
        if bet_won:
            # Win: Get back bet_size + profit
            # Profit = bet_size * (1/market_odds - 1)
            payout_multiplier = 1.0 / market_odds if market_odds > 0 else 0
            profit = bet_size * (payout_multiplier - 1.0)
            return profit
        else:
            # Loss: Lose the bet_size
            return -bet_size
    
    @staticmethod
    def evaluate_betting_results(
        betting_decisions: List[Dict],
        actual_temps_df: pd.DataFrame,
        starting_bankroll: float = 1000.0
    ) -> pd.DataFrame:
        """
        Evaluate all betting decisions and calculate outcomes and P&L.
        
        Args:
            betting_decisions: List of betting decision dictionaries
            actual_temps_df: DataFrame with actual temperature observations
                            Expected columns: 'timestamp', 'temperature_f'
            starting_bankroll: Starting bankroll in dollars
            
        Returns:
            DataFrame with bet-by-bet results including:
            - target_date
            - threshold
            - threshold_type
            - forecast_temp
            - actual_temp
            - bet_placed
            - bet_size
            - model_probability
            - market_odds
            - expected_value
            - bet_outcome (win/loss/no_bet)
            - profit_loss
            - cumulative_pl
        """
        results = []
        cumulative_pl = 0.0
        
        for decision in betting_decisions:
            target_date = decision['target_date']
            
            # Get actual temperature for the target date
            actual_temp = OutcomeEvaluator.get_actual_peak_temp(
                actual_temps_df,
                target_date
            )
            
            # Initialize result record
            result = {
                'target_date': target_date,
                'threshold': decision['threshold'],
                'threshold_type': decision['threshold_type'],
                'forecast_temp': decision['forecast_temp'],
                'actual_temp': actual_temp,
                'bet_placed': decision['should_bet'],
                'bet_size': decision['bet_size'],
                'model_probability': decision['model_probability'],
                'market_odds': decision['market_odds'],
                'expected_value': decision['expected_value'],
                'bet_outcome': 'no_bet',
                'profit_loss': 0.0,
                'cumulative_pl': cumulative_pl
            }
            
            # Evaluate outcome if bet was placed and we have actual temp
            if decision['should_bet'] and actual_temp is not None:
                bet_won = OutcomeEvaluator.evaluate_bet_outcome(
                    actual_temp,
                    decision['threshold_type'],
                    decision['threshold_low'],
                    decision['threshold_high']
                )
                
                profit_loss = OutcomeEvaluator.calculate_profit_loss(
                    decision['bet_size'],
                    bet_won,
                    decision['market_odds']
                )
                
                cumulative_pl += profit_loss
                
                result['bet_outcome'] = 'win' if bet_won else 'loss'
                result['profit_loss'] = profit_loss
                result['cumulative_pl'] = cumulative_pl
            
            results.append(result)
        
        return pd.DataFrame(results)
    
    @staticmethod
    def get_actual_peak_temp(
        actual_temps_df: pd.DataFrame,
        target_date: str
    ) -> Optional[float]:
        """
        Get the actual peak temperature for a target date.
        
        Args:
            actual_temps_df: DataFrame with actual temperature observations
                            Expected columns: 'timestamp', 'temperature_f'
            target_date: Target date in YYYY-MM-DD format
            
        Returns:
            Peak temperature in °F, or None if not found
        """
        # Convert timestamp to datetime if needed
        if 'timestamp' in actual_temps_df.columns:
            temps_df = actual_temps_df.copy()
            temps_df['timestamp'] = pd.to_datetime(temps_df['timestamp'])
            
            # Filter for target date
            target_dt = pd.to_datetime(target_date)
            date_temps = temps_df[temps_df['timestamp'].dt.date == target_dt.date()]
            
            if date_temps.empty:
                return None
            
            # Return maximum temperature
            return date_temps['temperature_f'].max()
        
        return None
    
    @staticmethod
    def generate_summary_statistics(results_df: pd.DataFrame) -> Dict:
        """
        Generate summary statistics from betting results.
        
        Args:
            results_df: DataFrame with betting results
            
        Returns:
            Dictionary with summary statistics
        """
        # Filter for bets that were actually placed
        bets_placed = results_df[results_df['bet_placed'] == True].copy()
        
        if bets_placed.empty:
            return {
                'total_bets': 0,
                'total_wagered': 0.0,
                'total_profit_loss': 0.0,
                'win_rate': 0.0,
                'roi': 0.0,
                'final_bankroll': 0.0
            }
        
        # Calculate statistics
        total_bets = len(bets_placed)
        total_wagered = bets_placed['bet_size'].sum()
        total_profit_loss = bets_placed['profit_loss'].sum()
        
        # Win rate (only for bets with outcomes)
        bets_with_outcome = bets_placed[bets_placed['bet_outcome'] != 'no_bet']
        if not bets_with_outcome.empty:
            wins = len(bets_with_outcome[bets_with_outcome['bet_outcome'] == 'win'])
            win_rate = wins / len(bets_with_outcome)
        else:
            win_rate = 0.0
        
        # ROI
        roi = (total_profit_loss / total_wagered * 100) if total_wagered > 0 else 0.0
        
        # Final cumulative P&L
        final_pl = bets_placed['cumulative_pl'].iloc[-1] if not bets_placed.empty else 0.0
        
        return {
            'total_bets': total_bets,
            'total_wagered': total_wagered,
            'total_profit_loss': total_profit_loss,
            'win_rate': win_rate,
            'roi': roi,
            'final_cumulative_pl': final_pl
        }


if __name__ == "__main__":
    # Example usage
    print("Betting Simulator Module")
    print("=" * 50)
    
    # Load data
    print("\nLoading data...")
    forecast_df = pd.read_csv('data/raw/historical_forecasts.csv')
    odds_df = pd.read_csv('data/raw/polymarket_odds_history.csv')
    
    # Initialize simulator
    simulator = BettingSimulator(
        ev_threshold=0.05,
        kelly_fraction=0.25,
        bankroll=1000.0,
        base_uncertainty=3.4
    )
    
    # Example: Simulate betting for a specific date
    target_date = "2025-02-01"
    
    # Get forecast
    forecast_temp = load_forecast_for_date(forecast_df, target_date)
    
    if forecast_temp is not None:
        print(f"\nForecast for {target_date}: {forecast_temp:.1f}°F")
        
        # Get market odds
        market_odds = load_market_odds_for_date(odds_df, target_date)
        
        if not market_odds.empty:
            print(f"Found {len(market_odds)} market thresholds")
            
            # Simulate betting decisions
            decisions = simulator.simulate_day_betting_decisions(
                target_date,
                forecast_temp,
                market_odds
            )
            
            # Display results
            print(f"\nBetting Decisions for {target_date}:")
            print("-" * 80)
            
            bets_placed = 0
            for decision in decisions:
                if decision['should_bet']:
                    bets_placed += 1
                    print(f"✓ BET PLACED - Threshold: {decision['threshold']}")
                    print(f"  Type: {decision['threshold_type']}")
                    print(f"  Forecast: {decision['forecast_temp']:.1f}°F")
                    print(f"  Model Prob: {decision['model_probability']:.2%}")
                    print(f"  Market Odds: {decision['market_odds']:.2%}")
                    print(f"  Expected Value: {decision['expected_value']:.2%}")
                    print(f"  Bet Size: ${decision['bet_size']:.2f}")
                    print()
            
            if bets_placed == 0:
                print("No bets placed (no positive EV opportunities)")
                print("\nAll evaluated thresholds:")
                for decision in decisions:
                    print(f"  {decision['threshold']}: "
                          f"Model={decision['model_probability']:.2%}, "
                          f"Market={decision['market_odds']:.2%}, "
                          f"EV={decision['expected_value']:.2%}")
        else:
            print(f"No market odds found for {target_date}")
    else:
        print(f"No forecast found for {target_date}")

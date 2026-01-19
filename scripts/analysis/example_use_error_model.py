"""
Example: How to use the saved error distribution results in future models
"""

import json
import numpy as np
from scipy import stats


def load_error_model():
    """Load the saved error distribution analysis"""
    with open('data/processed/error_distribution_analysis.json', 'r') as f:
        return json.load(f)


def calculate_probability_above_threshold(forecast_temp, threshold, error_model):
    """
    Calculate the probability that actual temperature will be above a threshold
    given a forecast temperature.
    
    Args:
        forecast_temp: Forecasted temperature (e.g., 45°F)
        threshold: Threshold temperature (e.g., 44°F for "≥44°F" bet)
        error_model: Loaded error distribution model
    
    Returns:
        Probability (0 to 1) that actual temp >= threshold
    """
    
    # Get error distribution parameters
    mean_error = error_model['basic_statistics']['mean']
    std_error = error_model['basic_statistics']['std']
    
    # Adjust forecast for bias
    # Forecast has +0.61°F warm bias, so actual is typically 0.61°F cooler
    adjusted_forecast = forecast_temp - mean_error
    
    # Calculate z-score
    # How many standard deviations is the threshold from our adjusted forecast?
    z_score = (threshold - adjusted_forecast) / std_error
    
    # Probability that actual >= threshold
    # This is 1 - CDF(z_score) because we want the upper tail
    prob = 1 - stats.norm.cdf(z_score)
    
    return prob


def get_confidence_interval(forecast_temp, confidence_level, error_model):
    """
    Get confidence interval for actual temperature given a forecast.
    
    Args:
        forecast_temp: Forecasted temperature
        confidence_level: Desired confidence (e.g., 0.68, 0.90, 0.95)
        error_model: Loaded error distribution model
    
    Returns:
        (lower_bound, upper_bound) tuple
    """
    
    mean_error = error_model['basic_statistics']['mean']
    std_error = error_model['basic_statistics']['std']
    
    # Adjust for bias
    adjusted_forecast = forecast_temp - mean_error
    
    # Calculate z-score for confidence level
    z = stats.norm.ppf((1 + confidence_level) / 2)
    
    # Calculate bounds
    margin = z * std_error
    lower = adjusted_forecast - margin
    upper = adjusted_forecast + margin
    
    return (lower, upper)


def example_betting_decision():
    """Example: Making a betting decision using the error model"""
    
    print("="*70)
    print("EXAMPLE: Using Error Model for Betting Decision")
    print("="*70)
    
    # Load error model
    error_model = load_error_model()
    
    print("\nError Model Summary:")
    print(f"  Mean Error (Bias): {error_model['basic_statistics']['mean']:+.2f}°F")
    print(f"  Std Deviation:     {error_model['basic_statistics']['std']:.2f}°F")
    print(f"  Is Normal:         {error_model['is_approximately_normal']}")
    
    # Example scenario
    forecast_temp = 45.0
    threshold = 44.0
    market_odds = 0.55  # Market says 55% chance of ≥44°F
    
    print("\n" + "="*70)
    print("BETTING SCENARIO:")
    print("="*70)
    print(f"\nForecast: {forecast_temp}°F (9 PM forecast for tomorrow)")
    print(f"Bet: Temperature will be ≥{threshold}°F")
    print(f"Market Odds: {market_odds:.0%} (implied probability)")
    
    # Calculate model probability
    model_prob = calculate_probability_above_threshold(forecast_temp, threshold, error_model)
    
    print(f"\n" + "="*70)
    print("MODEL ANALYSIS:")
    print("="*70)
    print(f"\nAdjusted Forecast: {forecast_temp - error_model['basic_statistics']['mean']:.1f}°F")
    print(f"  (Accounting for {error_model['basic_statistics']['mean']:+.2f}°F warm bias)")
    
    print(f"\nModel Probability: {model_prob:.1%}")
    print(f"Market Probability: {market_odds:.1%}")
    print(f"Edge: {(model_prob - market_odds):.1%}")
    
    # Calculate expected value
    if model_prob > market_odds:
        payout_multiplier = 1 / market_odds
        ev = (model_prob * payout_multiplier) - 1
        print(f"\nExpected Value: {ev:+.1%}")
        
        if ev > 0.05:  # 5% EV threshold
            print(f"✅ RECOMMENDATION: BET (EV > 5%)")
        else:
            print(f"⚠️  RECOMMENDATION: PASS (EV < 5%)")
    else:
        print(f"\n❌ RECOMMENDATION: NO BET (negative edge)")
    
    # Show confidence intervals
    print(f"\n" + "="*70)
    print("CONFIDENCE INTERVALS:")
    print("="*70)
    
    for conf_level in [0.68, 0.90, 0.95]:
        lower, upper = get_confidence_interval(forecast_temp, conf_level, error_model)
        print(f"\n{conf_level:.0%} Confidence: {lower:.1f}°F to {upper:.1f}°F")
        print(f"  Interpretation: {conf_level:.0%} chance actual temp is in this range")


def example_multiple_thresholds():
    """Example: Analyzing multiple betting thresholds"""
    
    print("\n\n" + "="*70)
    print("EXAMPLE: Analyzing Multiple Thresholds")
    print("="*70)
    
    error_model = load_error_model()
    forecast_temp = 45.0
    
    print(f"\nForecast: {forecast_temp}°F")
    print(f"\nProbabilities for different thresholds:")
    print("-"*70)
    print(f"{'Threshold':<15} {'Model Prob':<15} {'Interpretation'}")
    print("-"*70)
    
    thresholds = [40, 42, 44, 45, 46, 48, 50]
    for threshold in thresholds:
        prob = calculate_probability_above_threshold(forecast_temp, threshold, error_model)
        print(f"≥{threshold}°F{'':<10} {prob:>6.1%}{'':<9} ", end="")
        
        if prob > 0.75:
            print("Very likely")
        elif prob > 0.55:
            print("Likely")
        elif prob > 0.45:
            print("Toss-up")
        elif prob > 0.25:
            print("Unlikely")
        else:
            print("Very unlikely")


if __name__ == '__main__':
    example_betting_decision()
    example_multiple_thresholds()
    
    print("\n" + "="*70)
    print("These calculations use the saved error model from:")
    print("  data/processed/error_distribution_analysis.json")
    print("="*70)

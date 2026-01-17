"""
Betting strategy for Polymarket temperature markets.
Compares model probabilities to market odds to find +EV bets.
"""

import pandas as pd
import numpy as np
import pickle
from scipy import stats
from datetime import datetime, timedelta

def load_probabilistic_model(model_file='models/probabilistic_temp_model.pkl'):
    """Load trained quantile models."""
    with open(model_file, 'rb') as f:
        data = pickle.load(f)
    return data['models'], data['feature_cols'], data['quantiles']

def estimate_probability_above_threshold(models, features, threshold):
    """
    Estimate P(peak_temp > threshold) using quantile predictions.
    
    Args:
        models: Dict of quantile models
        features: Feature vector for prediction
        threshold: Temperature threshold
    
    Returns:
        Probability that peak temp exceeds threshold
    """
    # Get quantile predictions
    quantile_preds = {}
    for q, model in models.items():
        quantile_preds[q] = model.predict(features)[0]
    
    # Fit normal distribution to quantiles (simple approach)
    # Better: use actual quantile values to interpolate
    q50 = quantile_preds[0.5]  # median
    q10 = quantile_preds[0.1]
    q90 = quantile_preds[0.9]
    
    # Estimate std from quantiles (Q90 - Q10) / 2.56 ≈ std
    estimated_std = (q90 - q10) / 2.56
    
    # P(X > threshold) using normal approximation
    if estimated_std > 0:
        z_score = (threshold - q50) / estimated_std
        prob_above = 1 - stats.norm.cdf(z_score)
    else:
        prob_above = 1.0 if q50 > threshold else 0.0
    
    return prob_above, q50, estimated_std

def calculate_kelly_bet_size(prob_win, odds, bankroll, kelly_fraction=0.25):
    """
    Calculate optimal bet size using Kelly Criterion.
    
    Args:
        prob_win: Your estimated probability of winning
        odds: Market odds (e.g., 0.60 means 60% implied probability)
        bankroll: Total bankroll
        kelly_fraction: Fraction of Kelly to use (0.25 = quarter Kelly)
    
    Returns:
        Recommended bet size
    """
    # Convert market odds to decimal odds
    if odds >= 1.0 or odds <= 0:
        return 0
    
    decimal_odds = 1 / odds
    
    # Kelly formula: f = (bp - q) / b
    # where b = decimal_odds - 1, p = prob_win, q = 1 - prob_win
    b = decimal_odds - 1
    p = prob_win
    q = 1 - prob_win
    
    kelly = (b * p - q) / b
    
    # Apply fractional Kelly for safety
    kelly = max(0, kelly * kelly_fraction)
    
    bet_size = kelly * bankroll
    
    return bet_size

def analyze_market_opportunity(threshold, market_odds, model_prob, bankroll=1000):
    """
    Analyze if there's a +EV betting opportunity.
    
    Args:
        threshold: Temperature threshold (e.g., 75°F)
        market_odds: Market's implied probability (e.g., 0.40 = 40%)
        model_prob: Your model's probability
        bankroll: Your total bankroll
    
    Returns:
        Dict with analysis
    """
    # Expected value
    if market_odds >= 1.0 or market_odds <= 0:
        return None
    
    payout_multiplier = 1 / market_odds
    expected_value = (model_prob * payout_multiplier) - 1
    
    # Kelly bet size
    kelly_size = calculate_kelly_bet_size(model_prob, market_odds, bankroll)
    
    # Edge
    edge = model_prob - market_odds
    edge_pct = (edge / market_odds) * 100 if market_odds > 0 else 0
    
    analysis = {
        'threshold': threshold,
        'market_odds': market_odds,
        'model_prob': model_prob,
        'edge': edge,
        'edge_pct': edge_pct,
        'expected_value': expected_value,
        'ev_pct': expected_value * 100,
        'kelly_bet': kelly_size,
        'kelly_pct': (kelly_size / bankroll) * 100,
        'recommendation': 'BET' if expected_value > 0.05 else 'PASS'
    }
    
    return analysis

def print_betting_analysis(analysis):
    """Pretty print betting analysis."""
    if analysis is None:
        print("Invalid market odds")
        return
    
    print("\n" + "=" * 70)
    print("BETTING OPPORTUNITY ANALYSIS")
    print("=" * 70)
    print(f"\nMarket: Peak temp > {analysis['threshold']}°F")
    print(f"\nMarket Odds:     {analysis['market_odds']:.1%}")
    print(f"Model Estimate:  {analysis['model_prob']:.1%}")
    print(f"Edge:            {analysis['edge']:+.1%} ({analysis['edge_pct']:+.1f}%)")
    print(f"\nExpected Value:  {analysis['ev_pct']:+.2f}%")
    print(f"Kelly Bet Size:  ${analysis['kelly_bet']:.2f} ({analysis['kelly_pct']:.1f}% of bankroll)")
    print(f"\nRecommendation:  {analysis['recommendation']}")
    
    if analysis['recommendation'] == 'BET':
        print("\n✓ POSITIVE EXPECTED VALUE - Consider betting")
    else:
        print("\n✗ No edge - Pass on this market")
    
    print("=" * 70)

def example_analysis():
    """Example betting analysis."""
    
    print("=" * 70)
    print("Polymarket Temperature Betting Strategy")
    print("=" * 70)
    
    # Example: Analyze multiple thresholds
    print("\nExample Analysis (hypothetical data):")
    print("\nScenario: Tomorrow's forecast suggests 72°F peak")
    print("Model estimates: Q50=72°F, Q10=67°F, Q90=77°F")
    
    # Simulate model probabilities
    thresholds = [65, 70, 75, 80]
    market_odds = [0.85, 0.60, 0.25, 0.05]  # Market's implied probabilities
    model_probs = [0.92, 0.68, 0.18, 0.02]  # Your model's probabilities
    
    print("\n" + "-" * 70)
    for threshold, market, model in zip(thresholds, market_odds, model_probs):
        analysis = analyze_market_opportunity(threshold, market, model, bankroll=1000)
        
        print(f"\nTemp > {threshold}°F:")
        print(f"  Market: {market:.0%} | Model: {model:.0%} | Edge: {(model-market):+.0%}")
        print(f"  EV: {analysis['ev_pct']:+.1f}% | Kelly: ${analysis['kelly_bet']:.2f} | {analysis['recommendation']}")
    
    print("\n" + "-" * 70)
    print("\nKey Insights:")
    print("  • Look for markets where your model disagrees significantly (>5% edge)")
    print("  • Use fractional Kelly (25%) to manage risk")
    print("  • Only bet when EV > 5% to account for model uncertainty")
    print("  • Track results to calibrate your model over time")

if __name__ == "__main__":
    example_analysis()
    
    print("\n" + "=" * 70)
    print("Next Steps:")
    print("=" * 70)
    print("1. Train probabilistic model: python scripts/train_probabilistic_model.py")
    print("2. Fetch NWS forecast: python scripts/fetch_nws_forecast.py")
    print("3. Generate predictions with uncertainty estimates")
    print("4. Compare to Polymarket odds and identify +EV opportunities")

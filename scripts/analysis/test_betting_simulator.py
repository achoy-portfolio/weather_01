"""
Test script for betting simulator functionality
"""

import pandas as pd
import numpy as np
from betting_simulator import BettingSimulator


def test_parse_threshold():
    """Test threshold parsing"""
    print("Testing threshold parsing...")
    simulator = BettingSimulator()
    
    # Test range format
    result = simulator.parse_threshold("36-37")
    assert result == ("range", 36.0, 37.0), f"Expected ('range', 36.0, 37.0), got {result}"
    print("  ✓ Range format: 36-37")
    
    # Test above format
    result = simulator.parse_threshold("≥75")
    assert result == ("above", 75.0, None), f"Expected ('above', 75.0, None), got {result}"
    print("  ✓ Above format: ≥75")
    
    # Test below format
    result = simulator.parse_threshold("≤33")
    assert result == ("below", 33.0, None), f"Expected ('below', 33.0, None), got {result}"
    print("  ✓ Below format: ≤33")
    
    print("✓ All threshold parsing tests passed\n")


def test_model_probability():
    """Test model probability calculations"""
    print("Testing model probability calculations...")
    simulator = BettingSimulator(base_uncertainty=3.4)
    
    # Test "above" threshold
    # Forecast 50°F, threshold ≥50, should be ~50% (right at the mean)
    prob = simulator.calculate_model_probability(50.0, "above", 50.0, None, 3.4)
    assert 0.45 < prob < 0.55, f"Expected ~0.5, got {prob}"
    print(f"  ✓ Above threshold (at mean): {prob:.2%}")
    
    # Test "below" threshold
    # Forecast 50°F, threshold ≤50, should be ~50%
    prob = simulator.calculate_model_probability(50.0, "below", 50.0, None, 3.4)
    assert 0.45 < prob < 0.55, f"Expected ~0.5, got {prob}"
    print(f"  ✓ Below threshold (at mean): {prob:.2%}")
    
    # Test "range" threshold
    # Forecast 50°F, range 49-51, should be high probability
    prob = simulator.calculate_model_probability(50.0, "range", 49.0, 51.0, 3.4)
    assert prob > 0.3, f"Expected >0.3, got {prob}"
    print(f"  ✓ Range threshold (centered): {prob:.2%}")
    
    # Test extreme case - very unlikely
    # Forecast 50°F, threshold ≥70, should be very low
    prob = simulator.calculate_model_probability(50.0, "above", 70.0, None, 3.4)
    assert prob < 0.01, f"Expected <0.01, got {prob}"
    print(f"  ✓ Above threshold (far away): {prob:.2%}")
    
    print("✓ All probability calculation tests passed\n")


def test_expected_value():
    """Test expected value calculations"""
    print("Testing expected value calculations...")
    simulator = BettingSimulator()
    
    # Test positive EV
    # Model prob 60%, market odds 50% -> EV = (0.6 * 2) - 1 = 0.2 (20%)
    ev = simulator.calculate_expected_value(0.6, 0.5)
    assert abs(ev - 0.2) < 0.01, f"Expected 0.2, got {ev}"
    print(f"  ✓ Positive EV: {ev:.2%}")
    
    # Test negative EV
    # Model prob 40%, market odds 50% -> EV = (0.4 * 2) - 1 = -0.2 (-20%)
    ev = simulator.calculate_expected_value(0.4, 0.5)
    assert abs(ev - (-0.2)) < 0.01, f"Expected -0.2, got {ev}"
    print(f"  ✓ Negative EV: {ev:.2%}")
    
    # Test fair odds (no edge)
    # Model prob 50%, market odds 50% -> EV = 0
    ev = simulator.calculate_expected_value(0.5, 0.5)
    assert abs(ev) < 0.01, f"Expected ~0, got {ev}"
    print(f"  ✓ Fair odds (no edge): {ev:.2%}")
    
    print("✓ All expected value tests passed\n")


def test_kelly_criterion():
    """Test Kelly criterion bet sizing"""
    print("Testing Kelly criterion bet sizing...")
    simulator = BettingSimulator(kelly_fraction=0.25, bankroll=1000.0)
    
    # Test positive edge
    # Model prob 60%, market odds 50%, bankroll $1000
    # Full Kelly = (1*0.6 - 0.4) / 1 = 0.2 (20%)
    # Fractional Kelly (25%) = 0.05 (5%) = $50
    bet_size = simulator.calculate_kelly_bet_size(0.6, 0.5, 1000.0)
    assert 45 < bet_size < 55, f"Expected ~50, got {bet_size}"
    print(f"  ✓ Positive edge bet size: ${bet_size:.2f}")
    
    # Test no edge (should bet 0)
    bet_size = simulator.calculate_kelly_bet_size(0.5, 0.5, 1000.0)
    assert bet_size < 1, f"Expected ~0, got {bet_size}"
    print(f"  ✓ No edge bet size: ${bet_size:.2f}")
    
    # Test negative edge (should bet 0)
    bet_size = simulator.calculate_kelly_bet_size(0.4, 0.5, 1000.0)
    assert bet_size == 0, f"Expected 0, got {bet_size}"
    print(f"  ✓ Negative edge bet size: ${bet_size:.2f}")
    
    print("✓ All Kelly criterion tests passed\n")


def test_betting_decision():
    """Test complete betting decision logic"""
    print("Testing betting decision logic...")
    simulator = BettingSimulator(
        ev_threshold=0.05,
        kelly_fraction=0.25,
        bankroll=1000.0,
        base_uncertainty=3.4
    )
    
    # Test case 1: Should bet (good edge)
    # Forecast 50°F, threshold ≥45, market odds 30%
    # Model prob should be high (~93%), EV should be positive
    decision = simulator.should_place_bet(50.0, "≥45", 0.3)
    assert decision['should_bet'] == True, "Expected to place bet"
    assert decision['bet_size'] > 0, "Expected positive bet size"
    print(f"  ✓ Good edge scenario: Bet ${decision['bet_size']:.2f} (EV: {decision['expected_value']:.2%})")
    
    # Test case 2: Should not bet (no edge)
    # Forecast 50°F, threshold ≥50, market odds 50%
    # Model prob ~50%, EV ~0
    decision = simulator.should_place_bet(50.0, "≥50", 0.5)
    assert decision['should_bet'] == False, "Expected not to place bet"
    print(f"  ✓ No edge scenario: No bet (EV: {decision['expected_value']:.2%})")
    
    # Test case 3: Should not bet (negative edge)
    # Forecast 50°F, threshold ≥70, market odds 50%
    # Model prob very low, EV negative
    decision = simulator.should_place_bet(50.0, "≥70", 0.5)
    assert decision['should_bet'] == False, "Expected not to place bet"
    print(f"  ✓ Negative edge scenario: No bet (EV: {decision['expected_value']:.2%})")
    
    print("✓ All betting decision tests passed\n")


def test_simulate_day():
    """Test simulating a full day of betting decisions"""
    print("Testing full day simulation...")
    simulator = BettingSimulator()
    
    # Create mock market odds
    market_odds = pd.DataFrame({
        'threshold': ['≥50', '45-49', '≤44'],
        'yes_probability': [0.3, 0.4, 0.3]
    })
    
    # Simulate with forecast of 50°F
    decisions = simulator.simulate_day_betting_decisions(
        '2025-04-01',
        50.0,
        market_odds
    )
    
    assert len(decisions) == 3, f"Expected 3 decisions, got {len(decisions)}"
    assert all('target_date' in d for d in decisions), "Missing target_date in decisions"
    assert all('should_bet' in d for d in decisions), "Missing should_bet in decisions"
    
    print(f"  ✓ Generated {len(decisions)} betting decisions")
    print("✓ Full day simulation test passed\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Betting Simulator Test Suite")
    print("=" * 60)
    print()
    
    try:
        test_parse_threshold()
        test_model_probability()
        test_expected_value()
        test_kelly_criterion()
        test_betting_decision()
        test_simulate_day()
        
        print("=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

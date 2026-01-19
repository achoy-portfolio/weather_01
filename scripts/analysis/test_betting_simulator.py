"""
Test script for betting simulator functionality
"""

import pandas as pd
import numpy as np
from betting_simulator import BettingSimulator, OutcomeEvaluator


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


def test_polymarket_rounding():
    """Test Polymarket rounding rules"""
    print("Testing Polymarket rounding...")
    
    # Test standard rounding
    assert OutcomeEvaluator.apply_polymarket_rounding(34.5) == 35, "34.5 should round to 35"
    assert OutcomeEvaluator.apply_polymarket_rounding(34.4) == 34, "34.4 should round to 34"
    assert OutcomeEvaluator.apply_polymarket_rounding(35.0) == 35, "35.0 should round to 35"
    assert OutcomeEvaluator.apply_polymarket_rounding(34.6) == 35, "34.6 should round to 35"
    assert OutcomeEvaluator.apply_polymarket_rounding(34.49) == 34, "34.49 should round to 34"
    
    print("  ✓ All rounding tests passed")
    print("✓ Polymarket rounding test passed\n")


def test_bet_outcome_evaluation():
    """Test bet outcome evaluation for different threshold types"""
    print("Testing bet outcome evaluation...")
    
    # Test "above" threshold
    # Actual 50°F (rounds to 50), threshold ≥50 -> WIN
    result = OutcomeEvaluator.evaluate_bet_outcome(50.0, "above", 50.0)
    assert result == True, "50°F should win ≥50 bet"
    print("  ✓ Above threshold (at boundary): WIN")
    
    # Actual 49.5°F (rounds to 50), threshold ≥50 -> WIN
    result = OutcomeEvaluator.evaluate_bet_outcome(49.5, "above", 50.0)
    assert result == True, "49.5°F (rounds to 50) should win ≥50 bet"
    print("  ✓ Above threshold (rounds up): WIN")
    
    # Actual 49.4°F (rounds to 49), threshold ≥50 -> LOSS
    result = OutcomeEvaluator.evaluate_bet_outcome(49.4, "above", 50.0)
    assert result == False, "49.4°F (rounds to 49) should lose ≥50 bet"
    print("  ✓ Above threshold (rounds down): LOSS")
    
    # Test "below" threshold
    # Actual 33°F, threshold ≤33 -> WIN
    result = OutcomeEvaluator.evaluate_bet_outcome(33.0, "below", 33.0)
    assert result == True, "33°F should win ≤33 bet"
    print("  ✓ Below threshold (at boundary): WIN")
    
    # Actual 33.4°F (rounds to 33), threshold ≤33 -> WIN
    result = OutcomeEvaluator.evaluate_bet_outcome(33.4, "below", 33.0)
    assert result == True, "33.4°F (rounds to 33) should win ≤33 bet"
    print("  ✓ Below threshold (rounds down): WIN")
    
    # Actual 33.5°F (rounds to 34), threshold ≤33 -> LOSS
    result = OutcomeEvaluator.evaluate_bet_outcome(33.5, "below", 33.0)
    assert result == False, "33.5°F (rounds to 34) should lose ≤33 bet"
    print("  ✓ Below threshold (rounds up): LOSS")
    
    # Test "range" threshold
    # Actual 36.0°F, range 36-37 -> WIN
    result = OutcomeEvaluator.evaluate_bet_outcome(36.0, "range", 36.0, 37.0)
    assert result == True, "36°F should win 36-37 bet"
    print("  ✓ Range threshold (at lower bound): WIN")
    
    # Actual 37.0°F, range 36-37 -> WIN
    result = OutcomeEvaluator.evaluate_bet_outcome(37.0, "range", 36.0, 37.0)
    assert result == True, "37°F should win 36-37 bet"
    print("  ✓ Range threshold (at upper bound): WIN")
    
    # Actual 36.5°F (rounds to 37), range 36-37 -> WIN
    result = OutcomeEvaluator.evaluate_bet_outcome(36.5, "range", 36.0, 37.0)
    assert result == True, "36.5°F (rounds to 37) should win 36-37 bet"
    print("  ✓ Range threshold (within range): WIN")
    
    # Actual 35.4°F (rounds to 35), range 36-37 -> LOSS
    result = OutcomeEvaluator.evaluate_bet_outcome(35.4, "range", 36.0, 37.0)
    assert result == False, "35.4°F (rounds to 35) should lose 36-37 bet"
    print("  ✓ Range threshold (below range): LOSS")
    
    # Actual 37.5°F (rounds to 38), range 36-37 -> LOSS
    result = OutcomeEvaluator.evaluate_bet_outcome(37.5, "range", 36.0, 37.0)
    assert result == False, "37.5°F (rounds to 38) should lose 36-37 bet"
    print("  ✓ Range threshold (above range): LOSS")
    
    print("✓ All bet outcome evaluation tests passed\n")


def test_profit_loss_calculation():
    """Test profit/loss calculations"""
    print("Testing profit/loss calculations...")
    
    # Test winning bet
    # Bet $100 at 50% odds (2x payout) -> Win $100
    profit = OutcomeEvaluator.calculate_profit_loss(100.0, True, 0.5)
    assert abs(profit - 100.0) < 0.01, f"Expected $100 profit, got ${profit}"
    print(f"  ✓ Winning bet (50% odds): ${profit:.2f} profit")
    
    # Test winning bet with different odds
    # Bet $100 at 25% odds (4x payout) -> Win $300
    profit = OutcomeEvaluator.calculate_profit_loss(100.0, True, 0.25)
    assert abs(profit - 300.0) < 0.01, f"Expected $300 profit, got ${profit}"
    print(f"  ✓ Winning bet (25% odds): ${profit:.2f} profit")
    
    # Test losing bet
    # Bet $100, lose -> Lose $100
    profit = OutcomeEvaluator.calculate_profit_loss(100.0, False, 0.5)
    assert abs(profit - (-100.0)) < 0.01, f"Expected -$100 loss, got ${profit}"
    print(f"  ✓ Losing bet: ${profit:.2f} loss")
    
    print("✓ All profit/loss calculation tests passed\n")


def test_evaluate_betting_results():
    """Test full betting results evaluation"""
    print("Testing full betting results evaluation...")
    
    # Create mock betting decisions
    decisions = [
        {
            'target_date': '2025-01-21',
            'threshold': '≥20',
            'threshold_type': 'above',
            'threshold_low': 20.0,
            'threshold_high': None,
            'forecast_temp': 25.0,
            'should_bet': True,
            'bet_size': 50.0,
            'model_probability': 0.7,
            'market_odds': 0.5,
            'expected_value': 0.4
        },
        {
            'target_date': '2025-01-21',
            'threshold': '≤10',
            'threshold_type': 'below',
            'threshold_low': 10.0,
            'threshold_high': None,
            'forecast_temp': 25.0,
            'should_bet': True,
            'bet_size': 30.0,
            'model_probability': 0.1,
            'market_odds': 0.3,
            'expected_value': -0.67
        },
        {
            'target_date': '2025-01-21',
            'threshold': '≥50',
            'threshold_type': 'above',
            'threshold_low': 50.0,
            'threshold_high': None,
            'forecast_temp': 25.0,
            'should_bet': False,
            'bet_size': 0.0,
            'model_probability': 0.01,
            'market_odds': 0.2,
            'expected_value': -0.95
        }
    ]
    
    # Create mock actual temperatures (peak is 24.5°F, rounds to 25°F)
    actual_temps = pd.DataFrame({
        'timestamp': pd.date_range('2025-01-21', periods=24, freq='H'),
        'temperature_f': [20.0, 21.0, 22.0, 23.0, 24.5, 24.0, 23.0, 22.0,
                         21.0, 20.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0,
                         13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0]
    })
    
    # Evaluate results
    results_df = OutcomeEvaluator.evaluate_betting_results(
        decisions,
        actual_temps,
        starting_bankroll=1000.0
    )
    
    # Verify results
    assert len(results_df) == 3, f"Expected 3 results, got {len(results_df)}"
    
    # Check first bet (≥20, actual 24.5 rounds to 25) -> WIN
    bet1 = results_df.iloc[0]
    assert bet1['bet_placed'] == True, "First bet should be placed"
    assert bet1['bet_outcome'] == 'win', f"First bet should win, got {bet1['bet_outcome']}"
    assert bet1['profit_loss'] > 0, "First bet should have positive P&L"
    print(f"  ✓ Bet 1 (≥20): WIN, P&L=${bet1['profit_loss']:.2f}")
    
    # Check second bet (≤10, actual 24.5 rounds to 25) -> LOSS
    bet2 = results_df.iloc[1]
    assert bet2['bet_placed'] == True, "Second bet should be placed"
    assert bet2['bet_outcome'] == 'loss', f"Second bet should lose, got {bet2['bet_outcome']}"
    assert bet2['profit_loss'] < 0, "Second bet should have negative P&L"
    print(f"  ✓ Bet 2 (≤10): LOSS, P&L=${bet2['profit_loss']:.2f}")
    
    # Check third bet (not placed)
    bet3 = results_df.iloc[2]
    assert bet3['bet_placed'] == False, "Third bet should not be placed"
    assert bet3['bet_outcome'] == 'no_bet', "Third bet should have no outcome"
    assert bet3['profit_loss'] == 0, "Third bet should have zero P&L"
    print(f"  ✓ Bet 3 (≥50): NO BET")
    
    # Check cumulative P&L
    assert bet1['cumulative_pl'] == bet1['profit_loss'], "First cumulative P&L should equal first P&L"
    assert bet2['cumulative_pl'] == bet1['profit_loss'] + bet2['profit_loss'], "Second cumulative P&L incorrect"
    print(f"  ✓ Cumulative P&L: ${bet2['cumulative_pl']:.2f}")
    
    print("✓ All betting results evaluation tests passed\n")


def test_summary_statistics():
    """Test summary statistics generation"""
    print("Testing summary statistics...")
    
    # Create mock results
    results_df = pd.DataFrame({
        'target_date': ['2025-01-21', '2025-01-21', '2025-01-22'],
        'bet_placed': [True, True, False],
        'bet_size': [100.0, 50.0, 0.0],
        'bet_outcome': ['win', 'loss', 'no_bet'],
        'profit_loss': [100.0, -50.0, 0.0],
        'cumulative_pl': [100.0, 50.0, 50.0]
    })
    
    summary = OutcomeEvaluator.generate_summary_statistics(results_df)
    
    assert summary['total_bets'] == 2, f"Expected 2 bets, got {summary['total_bets']}"
    assert summary['total_wagered'] == 150.0, f"Expected $150 wagered, got ${summary['total_wagered']}"
    assert summary['total_profit_loss'] == 50.0, f"Expected $50 P&L, got ${summary['total_profit_loss']}"
    assert abs(summary['win_rate'] - 0.5) < 0.01, f"Expected 50% win rate, got {summary['win_rate']}"
    assert abs(summary['roi'] - 33.33) < 0.1, f"Expected 33.33% ROI, got {summary['roi']}"
    
    print(f"  ✓ Total bets: {summary['total_bets']}")
    print(f"  ✓ Total wagered: ${summary['total_wagered']:.2f}")
    print(f"  ✓ Total P&L: ${summary['total_profit_loss']:.2f}")
    print(f"  ✓ Win rate: {summary['win_rate']:.1%}")
    print(f"  ✓ ROI: {summary['roi']:.1f}%")
    
    print("✓ All summary statistics tests passed\n")


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
        test_polymarket_rounding()
        test_bet_outcome_evaluation()
        test_profit_loss_calculation()
        test_evaluate_betting_results()
        test_summary_statistics()
        
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

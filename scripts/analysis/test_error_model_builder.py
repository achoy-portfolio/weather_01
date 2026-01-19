"""
Tests for error model builder
"""

import pytest
import pandas as pd
import numpy as np
from error_model_builder import (
    calculate_forecast_error,
    calculate_mae,
    calculate_rmse,
    calculate_bias,
    calculate_std_dev,
    get_season,
    calculate_lead_time
)


def test_calculate_forecast_error():
    """Test forecast error calculation (forecast - actual)"""
    assert calculate_forecast_error(50.0, 45.0) == 5.0
    assert calculate_forecast_error(45.0, 50.0) == -5.0
    assert calculate_forecast_error(50.0, 50.0) == 0.0


def test_calculate_mae():
    """Test Mean Absolute Error calculation"""
    errors = np.array([2.0, -3.0, 1.0, -1.0])
    mae = calculate_mae(errors)
    assert mae == 1.75  # (2 + 3 + 1 + 1) / 4


def test_calculate_rmse():
    """Test Root Mean Squared Error calculation"""
    errors = np.array([2.0, -2.0, 1.0, -1.0])
    rmse = calculate_rmse(errors)
    expected = np.sqrt((4 + 4 + 1 + 1) / 4)  # sqrt(10/4) = sqrt(2.5)
    assert abs(rmse - expected) < 0.001


def test_calculate_bias():
    """Test bias calculation (mean error)"""
    errors = np.array([2.0, -3.0, 1.0, -1.0])
    bias = calculate_bias(errors)
    assert bias == -0.25  # (2 - 3 + 1 - 1) / 4
    
    # Test positive bias (over-prediction)
    errors_positive = np.array([2.0, 3.0, 1.0, 1.0])
    assert calculate_bias(errors_positive) == 1.75
    
    # Test negative bias (under-prediction)
    errors_negative = np.array([-2.0, -3.0, -1.0, -1.0])
    assert calculate_bias(errors_negative) == -1.75


def test_calculate_std_dev():
    """Test standard deviation calculation"""
    errors = np.array([2.0, 4.0, 6.0, 8.0])
    std_dev = calculate_std_dev(errors)
    expected = np.std(errors)
    assert abs(std_dev - expected) < 0.001


def test_get_season():
    """Test season determination from month"""
    # Winter
    assert get_season(12) == 'winter'
    assert get_season(1) == 'winter'
    assert get_season(2) == 'winter'
    
    # Spring
    assert get_season(3) == 'spring'
    assert get_season(4) == 'spring'
    assert get_season(5) == 'spring'
    
    # Summer
    assert get_season(6) == 'summer'
    assert get_season(7) == 'summer'
    assert get_season(8) == 'summer'
    
    # Fall
    assert get_season(9) == 'fall'
    assert get_season(10) == 'fall'
    assert get_season(11) == 'fall'


def test_calculate_lead_time():
    """Test lead time calculation"""
    # 1-day lead time
    forecast_date = pd.to_datetime('2025-01-20')
    target_date = pd.to_datetime('2025-01-21')
    assert calculate_lead_time(forecast_date, target_date) == 1
    
    # 2-day lead time
    forecast_date = pd.to_datetime('2025-01-20')
    target_date = pd.to_datetime('2025-01-22')
    assert calculate_lead_time(forecast_date, target_date) == 2
    
    # Same day (0-day lead time)
    forecast_date = pd.to_datetime('2025-01-20')
    target_date = pd.to_datetime('2025-01-20')
    assert calculate_lead_time(forecast_date, target_date) == 0
    
    # Test with string inputs
    assert calculate_lead_time('2025-01-20', '2025-01-21') == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

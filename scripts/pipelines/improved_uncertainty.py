"""
Improved uncertainty estimation for Polymarket betting.
Handles multiple forecast sources and dynamic uncertainty.
"""

import numpy as np
import pandas as pd
from datetime import datetime

class ForecastEnsemble:
    """
    Ensemble multiple forecast sources with uncertainty estimation.
    """
    
    # Typical MAE for each source (from verification studies)
    SOURCE_UNCERTAINTY = {
        'nws': 2.5,           # NWS day-1 forecast
        'visual_crossing': 2.6,  # Visual Crossing
        'open_meteo': 2.5,    # Open-Meteo ECMWF
        'ml_model': 4.0,      # Your ML model without forecast
        'ml_with_forecast': 3.0  # Your ML model with forecast features
    }
    
    def __init__(self):
        self.predictions = {}
        self.uncertainties = {}
        
    def add_forecast(self, source, value, uncertainty=None):
        """
        Add a forecast from a source.
        
        Args:
            source: Source name (e.g., 'nws', 'visual_crossing')
            value: Predicted temperature
            uncertainty: Optional custom uncertainty (uses default if None)
        """
        if value is None:
            return
            
        self.predictions[source] = value
        
        if uncertainty is None:
            uncertainty = self.SOURCE_UNCERTAINTY.get(source, 5.0)
        
        self.uncertainties[source] = uncertainty
    
    def get_agreement_metrics(self):
        """Calculate agreement between forecasts."""
        if len(self.predictions) < 2:
            return {
                'spread': None,
                'std': None,
                'agreement': 'insufficient_data'
            }
        
        values = list(self.predictions.values())
        spread = max(values) - min(values)
        std = np.std(values)
        
        # Classify agreement
        if spread < 2:
            agreement = 'strong'
        elif spread < 4:
            agreement = 'moderate'
        elif spread < 6:
            agreement = 'weak'
        else:
            agreement = 'poor'
        
        return {
            'spread': spread,
            'std': std,
            'agreement': agreement,
            'n_forecasts': len(self.predictions)
        }
    
    def get_consensus(self, method='inverse_variance'):
        """
        Get consensus forecast and uncertainty.
        
        Args:
            method: 'simple', 'weighted', or 'inverse_variance'
        
        Returns:
            (consensus_value, consensus_uncertainty)
        """
        if len(self.predictions) == 0:
            return None, None
        
        if len(self.predictions) == 1:
            source = list(self.predictions.keys())[0]
            return self.predictions[source], self.uncertainties[source]
        
        if method == 'simple':
            # Simple average
            consensus = np.mean(list(self.predictions.values()))
            # Use average uncertainty
            uncertainty = np.mean(list(self.uncertainties.values()))
            
        elif method == 'weighted':
            # Predefined weights based on reliability
            weights = {
                'nws': 0.50,
                'visual_crossing': 0.25,
                'open_meteo': 0.25,
                'ml_model': 0.15,
                'ml_with_forecast': 0.25
            }
            
            total_weight = sum(weights.get(src, 0.2) for src in self.predictions)
            
            consensus = sum(
                weights.get(src, 0.2) * val 
                for src, val in self.predictions.items()
            ) / total_weight
            
            # Weighted uncertainty
            uncertainty = sum(
                weights.get(src, 0.2) * self.uncertainties[src]
                for src in self.predictions
            ) / total_weight
            
        elif method == 'inverse_variance':
            # Inverse variance weighting (optimal for independent forecasts)
            precisions = {
                src: 1 / (unc ** 2)
                for src, unc in self.uncertainties.items()
            }
            
            total_precision = sum(precisions.values())
            
            consensus = sum(
                precisions[src] * self.predictions[src]
                for src in self.predictions
            ) / total_precision
            
            # Combined uncertainty (precision-weighted)
            uncertainty = 1 / np.sqrt(total_precision)
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Adjust uncertainty based on agreement
        metrics = self.get_agreement_metrics()
        
        if metrics['agreement'] == 'poor':
            # Forecasts disagree - increase uncertainty
            uncertainty *= 1.5
        elif metrics['agreement'] == 'weak':
            uncertainty *= 1.2
        elif metrics['agreement'] == 'strong':
            # Forecasts agree - slightly reduce uncertainty
            uncertainty *= 0.95
        
        return consensus, uncertainty
    
    def should_bet(self, min_agreement='moderate'):
        """
        Check if forecast agreement is sufficient for betting.
        
        Args:
            min_agreement: Minimum required agreement level
        
        Returns:
            (should_bet, reason)
        """
        metrics = self.get_agreement_metrics()
        
        if metrics['agreement'] == 'insufficient_data':
            return False, "Need at least 2 forecasts"
        
        agreement_levels = ['poor', 'weak', 'moderate', 'strong']
        
        if agreement_levels.index(metrics['agreement']) < agreement_levels.index(min_agreement):
            return False, f"Forecast agreement is {metrics['agreement']} (spread: {metrics['spread']:.1f}°F)"
        
        return True, f"Forecast agreement is {metrics['agreement']}"
    
    def get_summary(self):
        """Get summary of all forecasts."""
        if len(self.predictions) == 0:
            return "No forecasts available"
        
        lines = ["Forecast Summary:"]
        lines.append("-" * 50)
        
        for source, value in self.predictions.items():
            unc = self.uncertainties[source]
            lines.append(f"  {source:20s}: {value:5.1f}°F ± {unc:.1f}°F")
        
        lines.append("-" * 50)
        
        consensus, unc = self.get_consensus()
        metrics = self.get_agreement_metrics()
        
        lines.append(f"  Consensus:           {consensus:5.1f}°F ± {unc:.1f}°F")
        lines.append(f"  Agreement:           {metrics['agreement']} (spread: {metrics['spread']:.1f}°F)")
        
        should_bet, reason = self.should_bet()
        lines.append(f"  Betting confidence:  {'✓' if should_bet else '✗'} {reason}")
        
        return "\n".join(lines)


def calculate_dynamic_uncertainty(
    nws_forecast=None,
    vc_forecast=None, 
    ml_prediction=None,
    have_todays_temps=False,
    rapid_change=False,
    extreme_temp=False
):
    """
    Calculate uncertainty dynamically based on conditions.
    
    Args:
        nws_forecast: NWS forecast value
        vc_forecast: Visual Crossing forecast value
        ml_prediction: ML model prediction
        have_todays_temps: Whether we have today's intraday temps
        rapid_change: Whether temps are changing rapidly (>15°F day-to-day)
        extreme_temp: Whether forecast is extreme (<10°F or >95°F)
    
    Returns:
        uncertainty in °F
    """
    ensemble = ForecastEnsemble()
    
    if nws_forecast is not None:
        ensemble.add_forecast('nws', nws_forecast)
    
    if vc_forecast is not None:
        ensemble.add_forecast('visual_crossing', vc_forecast)
    
    if ml_prediction is not None:
        # Use better uncertainty if we have forecast features
        if nws_forecast is not None or vc_forecast is not None:
            ensemble.add_forecast('ml_with_forecast', ml_prediction)
        else:
            ensemble.add_forecast('ml_model', ml_prediction)
    
    # Get consensus uncertainty
    _, uncertainty = ensemble.get_consensus(method='inverse_variance')
    
    if uncertainty is None:
        # No forecasts available - use conservative default
        uncertainty = 6.0
    
    # Adjust for conditions
    if have_todays_temps:
        # Having today's temps reduces uncertainty
        uncertainty *= 0.85
    
    if rapid_change:
        # Rapid changes increase uncertainty
        uncertainty *= 1.5
    
    if extreme_temp:
        # Extreme temps are harder to predict
        uncertainty *= 1.3
    
    return uncertainty


def get_minimum_edge_required(uncertainty):
    """
    Calculate minimum edge required based on uncertainty.
    Higher uncertainty requires larger edge.
    
    Args:
        uncertainty: Prediction uncertainty in °F
    
    Returns:
        Minimum edge required (as decimal, e.g., 0.05 = 5%)
    """
    # Base requirement
    base_edge = 0.05  # 5%
    
    # Increase requirement for high uncertainty
    if uncertainty > 5:
        return 0.10  # 10% edge required
    elif uncertainty > 4:
        return 0.08  # 8% edge required
    elif uncertainty > 3:
        return 0.06  # 6% edge required
    else:
        return base_edge


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("FORECAST ENSEMBLE EXAMPLE")
    print("=" * 70)
    print()
    
    # Create ensemble
    ensemble = ForecastEnsemble()
    
    # Add forecasts
    ensemble.add_forecast('nws', 72.0)
    ensemble.add_forecast('visual_crossing', 71.5)
    ensemble.add_forecast('ml_with_forecast', 73.0)
    
    # Get summary
    print(ensemble.get_summary())
    print()
    
    # Get consensus
    consensus, uncertainty = ensemble.get_consensus(method='inverse_variance')
    print(f"\nConsensus Prediction: {consensus:.1f}°F ± {uncertainty:.1f}°F")
    
    # Check if should bet
    should_bet, reason = ensemble.should_bet(min_agreement='moderate')
    print(f"Should bet: {should_bet} - {reason}")
    
    # Get minimum edge required
    min_edge = get_minimum_edge_required(uncertainty)
    print(f"Minimum edge required: {min_edge:.1%}")
    
    print("\n" + "=" * 70)
    print("SCENARIO: FORECASTS DISAGREE")
    print("=" * 70)
    print()
    
    # Create ensemble with disagreement
    ensemble2 = ForecastEnsemble()
    ensemble2.add_forecast('nws', 72.0)
    ensemble2.add_forecast('visual_crossing', 68.0)  # Disagrees by 4°F
    ensemble2.add_forecast('ml_with_forecast', 70.0)
    
    print(ensemble2.get_summary())
    
    consensus2, uncertainty2 = ensemble2.get_consensus(method='inverse_variance')
    print(f"\nConsensus Prediction: {consensus2:.1f}°F ± {uncertainty2:.1f}°F")
    print(f"Note: Uncertainty increased due to disagreement")
    
    should_bet2, reason2 = ensemble2.should_bet(min_agreement='moderate')
    print(f"Should bet: {should_bet2} - {reason2}")


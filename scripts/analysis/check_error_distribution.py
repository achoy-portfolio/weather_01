"""
Check if forecast errors are normally distributed
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def analyze_error_distribution():
    """Analyze the distribution of forecast errors"""
    
    print("="*70)
    print("CHECKING IF FORECAST ERRORS ARE NORMALLY DISTRIBUTED")
    print("="*70)
    
    # Load forecasts and actuals
    forecasts = pd.read_csv('data/raw/historical_forecasts.csv')
    forecasts['forecast_datetime'] = pd.to_datetime(
        forecasts['forecast_date'] + ' ' + forecasts['forecast_time']
    )
    forecasts['valid_time'] = pd.to_datetime(forecasts['valid_time'])
    
    actuals = pd.read_csv('data/raw/actual_temperatures.csv')
    actuals['timestamp'] = pd.to_datetime(actuals['timestamp'])
    actuals.rename(columns={'temperature_f': 'actual_temp'}, inplace=True)
    
    # Get 9 PM forecasts for next day
    results = []
    forecast_dates = forecasts['forecast_date'].unique()
    
    for forecast_date in forecast_dates:
        evening_forecast = forecasts[
            (forecasts['forecast_date'] == forecast_date) &
            (forecasts['forecast_time'] == '21:00')
        ].copy()
        
        if len(evening_forecast) == 0:
            continue
        
        next_day = pd.to_datetime(forecast_date) + pd.Timedelta(days=1)
        next_day_forecasts = evening_forecast[
            evening_forecast['valid_time'].dt.date == next_day.date()
        ]
        
        if len(next_day_forecasts) == 0:
            continue
        
        forecasted_max = next_day_forecasts['temperature'].max()
        
        next_day_actuals = actuals[
            actuals['timestamp'].dt.date == next_day.date()
        ]
        
        if len(next_day_actuals) == 0:
            continue
        
        actual_max = next_day_actuals['actual_temp'].max()
        error = forecasted_max - actual_max
        
        results.append(error)
    
    errors = np.array(results)
    
    print(f"\nAnalyzing {len(errors)} forecast errors...")
    
    # Basic statistics
    print(f"\nBasic Statistics:")
    print(f"  Mean:   {np.mean(errors):.2f}°F")
    print(f"  Median: {np.median(errors):.2f}°F")
    print(f"  Std:    {np.std(errors):.2f}°F")
    print(f"  Min:    {np.min(errors):.2f}°F")
    print(f"  Max:    {np.max(errors):.2f}°F")
    
    # Percentiles
    print(f"\nPercentiles:")
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    for p in percentiles:
        val = np.percentile(errors, p)
        print(f"  {p:2d}th: {val:+6.2f}°F")
    
    # Test for normality
    print(f"\n" + "="*70)
    print("NORMALITY TESTS:")
    print("="*70)
    
    # Shapiro-Wilk test
    statistic, p_value = stats.shapiro(errors)
    print(f"\nShapiro-Wilk Test:")
    print(f"  Statistic: {statistic:.4f}")
    print(f"  P-value:   {p_value:.4f}")
    if p_value < 0.05:
        print(f"  Result: ❌ NOT normally distributed (p < 0.05)")
    else:
        print(f"  Result: ✅ Could be normally distributed (p >= 0.05)")
    
    # Kolmogorov-Smirnov test
    statistic, p_value = stats.kstest(errors, 'norm', args=(np.mean(errors), np.std(errors)))
    print(f"\nKolmogorov-Smirnov Test:")
    print(f"  Statistic: {statistic:.4f}")
    print(f"  P-value:   {p_value:.4f}")
    if p_value < 0.05:
        print(f"  Result: ❌ NOT normally distributed (p < 0.05)")
    else:
        print(f"  Result: ✅ Could be normally distributed (p >= 0.05)")
    
    # Skewness and Kurtosis
    skewness = stats.skew(errors)
    kurtosis = stats.kurtosis(errors)
    print(f"\nShape Statistics:")
    print(f"  Skewness: {skewness:.2f} (0 = symmetric, + = right tail, - = left tail)")
    print(f"  Kurtosis: {kurtosis:.2f} (0 = normal, + = heavy tails, - = light tails)")
    
    # Calculate actual percentages within 1, 2, 3 standard deviations
    std = np.std(errors)
    mean = np.mean(errors)
    
    print(f"\n" + "="*70)
    print("ACTUAL vs NORMAL DISTRIBUTION:")
    print("="*70)
    
    within_1std = np.sum(np.abs(errors - mean) <= std) / len(errors) * 100
    within_2std = np.sum(np.abs(errors - mean) <= 2*std) / len(errors) * 100
    within_3std = np.sum(np.abs(errors - mean) <= 3*std) / len(errors) * 100
    
    print(f"\nWithin ±1 std ({std:.2f}°F):")
    print(f"  Actual:   {within_1std:.1f}%")
    print(f"  Normal:   68.3%")
    print(f"  Match:    {'✅' if abs(within_1std - 68.3) < 5 else '❌'}")
    
    print(f"\nWithin ±2 std ({2*std:.2f}°F):")
    print(f"  Actual:   {within_2std:.1f}%")
    print(f"  Normal:   95.4%")
    print(f"  Match:    {'✅' if abs(within_2std - 95.4) < 5 else '❌'}")
    
    print(f"\nWithin ±3 std ({3*std:.2f}°F):")
    print(f"  Actual:   {within_3std:.1f}%")
    print(f"  Normal:   99.7%")
    print(f"  Match:    {'✅' if abs(within_3std - 99.7) < 5 else '❌'}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Histogram with normal curve overlay
    ax = axes[0, 0]
    ax.hist(errors, bins=30, density=True, alpha=0.7, edgecolor='black')
    x = np.linspace(errors.min(), errors.max(), 100)
    ax.plot(x, stats.norm.pdf(x, mean, std), 'r-', linewidth=2, label='Normal Distribution')
    ax.axvline(mean, color='blue', linestyle='--', linewidth=2, label=f'Mean: {mean:.2f}°F')
    ax.set_xlabel('Forecast Error (°F)')
    ax.set_ylabel('Density')
    ax.set_title('Error Distribution vs Normal')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Q-Q plot
    ax = axes[0, 1]
    stats.probplot(errors, dist="norm", plot=ax)
    ax.set_title('Q-Q Plot (Normal Distribution)')
    ax.grid(True, alpha=0.3)
    
    # Box plot
    ax = axes[1, 0]
    ax.boxplot(errors, vert=True)
    ax.set_ylabel('Forecast Error (°F)')
    ax.set_title('Box Plot of Errors')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='red', linestyle='--', alpha=0.5)
    
    # Cumulative distribution
    ax = axes[1, 1]
    sorted_errors = np.sort(errors)
    cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
    ax.plot(sorted_errors, cumulative, 'b-', linewidth=2, label='Actual')
    
    # Normal CDF
    x_norm = np.linspace(errors.min(), errors.max(), 100)
    cdf_norm = stats.norm.cdf(x_norm, mean, std) * 100
    ax.plot(x_norm, cdf_norm, 'r--', linewidth=2, label='Normal')
    
    ax.set_xlabel('Forecast Error (°F)')
    ax.set_ylabel('Cumulative Probability (%)')
    ax.set_title('Cumulative Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('forecast_error_distribution.png', dpi=150)
    print(f"\n" + "="*70)
    print(f"Visualization saved to: forecast_error_distribution.png")
    print(f"="*70)
    
    # Save results to JSON
    results = {
        'created_at': pd.Timestamp.now().isoformat(),
        'sample_size': int(len(errors)),
        'basic_statistics': {
            'mean': float(np.mean(errors)),
            'median': float(np.median(errors)),
            'std': float(np.std(errors)),
            'min': float(np.min(errors)),
            'max': float(np.max(errors))
        },
        'percentiles': {
            f'p{p}': float(np.percentile(errors, p))
            for p in [5, 10, 25, 50, 75, 90, 95]
        },
        'normality_tests': {
            'shapiro_wilk': {
                'statistic': float(statistic),
                'p_value': float(p_value),
                'is_normal': bool(p_value >= 0.05)
            },
            'skewness': float(skewness),
            'kurtosis': float(kurtosis)
        },
        'distribution_fit': {
            'within_1std': {
                'actual_pct': float(within_1std),
                'expected_pct': 68.3,
                'std_value': float(std),
                'matches': bool(abs(within_1std - 68.3) < 5)
            },
            'within_2std': {
                'actual_pct': float(within_2std),
                'expected_pct': 95.4,
                'std_value': float(2*std),
                'matches': bool(abs(within_2std - 95.4) < 5)
            },
            'within_3std': {
                'actual_pct': float(within_3std),
                'expected_pct': 99.7,
                'std_value': float(3*std),
                'matches': bool(abs(within_3std - 99.7) < 5)
            }
        },
        'confidence_intervals': {
            '50_pct': {
                'range': float(np.percentile(np.abs(errors), 50)),
                'description': '50% of forecasts within this range'
            },
            '68_pct': {
                'range': float(np.percentile(np.abs(errors), 68)),
                'description': '68% of forecasts within this range'
            },
            '90_pct': {
                'range': float(np.percentile(np.abs(errors), 90)),
                'description': '90% of forecasts within this range'
            },
            '95_pct': {
                'range': float(np.percentile(np.abs(errors), 95)),
                'description': '95% of forecasts within this range'
            }
        },
        'is_approximately_normal': bool(abs(within_1std - 68.3) < 5 and abs(within_2std - 95.4) < 5),
        'recommendation': 'Use normal distribution assumptions' if abs(within_1std - 68.3) < 5 else 'Use empirical percentiles'
    }
    
    import json
    output_path = 'data/processed/error_distribution_analysis.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {output_path}")
    
    # Conclusion
    print(f"\n" + "="*70)
    print("CONCLUSION:")
    print("="*70)
    
    if results['is_approximately_normal']:
        print("\n✅ Errors appear to be approximately normally distributed")
        print("   You can use ±1 std (~68%) and ±2 std (~95%) confidence intervals")
    else:
        print("\n⚠️  Errors may NOT be perfectly normally distributed")
        print("   Use actual percentiles instead of assuming normal distribution")
        print(f"\n   Better to say:")
        print(f"   - 50% of forecasts within ±{results['confidence_intervals']['50_pct']['range']:.2f}°F")
        print(f"   - 68% of forecasts within ±{results['confidence_intervals']['68_pct']['range']:.2f}°F")
        print(f"   - 90% of forecasts within ±{results['confidence_intervals']['90_pct']['range']:.2f}°F")
        print(f"   - 95% of forecasts within ±{results['confidence_intervals']['95_pct']['range']:.2f}°F")
    
    return results


if __name__ == '__main__':
    results = analyze_error_distribution()
    
    print("\n" + "="*70)
    print("SAVED RESULTS FOR FUTURE USE:")
    print("="*70)
    print("\nFiles created:")
    print("  1. data/processed/error_distribution_analysis.json")
    print("     - Complete statistical analysis")
    print("     - Confidence intervals")
    print("     - Normality test results")
    print("\n  2. forecast_error_distribution.png")
    print("     - Visual analysis of error distribution")
    print("\nThese files can be loaded by future models to:")
    print("  - Calculate betting probabilities")
    print("  - Estimate uncertainty")
    print("  - Set confidence intervals")

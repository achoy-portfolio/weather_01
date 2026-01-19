"""
Check if forecast errors are normally distributed

Updated to use latest forecast accuracy analysis findings and analyze by lead time.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import json
from pathlib import Path


def load_forecast_data():
    """Load forecast and actual data using same logic as forecast_accuracy_analysis"""
    print("Loading data...")
    
    # Load forecasts
    forecast_files = [
        'data/raw/historical_forecasts.csv',
        'data/raw/openmeteo_previous_runs.csv'
    ]
    
    forecasts = None
    for file_path in forecast_files:
        if Path(file_path).exists():
            forecasts = pd.read_csv(file_path)
            print(f"Loaded forecasts from: {file_path}")
            break
    
    if forecasts is None:
        raise FileNotFoundError("No forecast data found")
    
    # Parse timestamps and normalize column names
    forecasts['valid_time'] = pd.to_datetime(forecasts['valid_time'])
    
    # Handle different CSV formats
    if 'lead_time' in forecasts.columns:
        forecasts['forecast_issued'] = pd.to_datetime(forecasts['forecast_issued'])
    elif 'days_before' in forecasts.columns:
        forecasts['lead_time'] = forecasts['days_before']
        forecasts['forecast_issued'] = pd.to_datetime(
            forecasts['forecast_date'] + ' ' + forecasts['forecast_time']
        )
    
    # Load actuals
    try:
        actuals = pd.read_csv('data/raw/wunderground_hourly_temps.csv')
        actuals['timestamp'] = pd.to_datetime(actuals['timestamp'])
        actuals.rename(columns={'temperature_f': 'actual_temp'}, inplace=True)
        print("Using Weather Underground hourly data")
    except FileNotFoundError:
        actuals = pd.DataFrame(columns=['timestamp', 'actual_temp'])
        print("WARNING: No actual temperature data found")
    
    # Load daily max
    try:
        daily_max = pd.read_csv('data/raw/wunderground_daily_max_temps.csv')
        daily_max['date'] = pd.to_datetime(daily_max['date'])
        print(f"Loaded {len(daily_max)} days of daily max temperatures")
    except FileNotFoundError:
        daily_max = None
        print("No daily max file found")
    
    return forecasts, actuals, daily_max
def calculate_daily_max_errors(forecasts, daily_max):
    """Calculate errors for daily maximum temperature predictions by lead time"""
    
    if daily_max is None:
        return None
    
    # Calculate forecasted daily max for each target date and lead time
    forecasts_copy = forecasts.copy()
    forecasts_copy['target_date'] = forecasts_copy['valid_time'].dt.date
    
    # Group by target_date and lead_time, get max temperature
    forecast_daily_max = forecasts_copy.groupby(['target_date', 'lead_time']).agg({
        'temperature': 'max'
    }).reset_index()
    forecast_daily_max.rename(columns={'temperature': 'forecasted_max'}, inplace=True)
    
    # Prepare actual daily max
    actual_daily_max = daily_max.copy()
    actual_daily_max['date'] = actual_daily_max['date'].dt.date
    
    # Merge
    merged = forecast_daily_max.merge(
        actual_daily_max,
        left_on='target_date',
        right_on='date',
        how='inner'
    )
    
    # Calculate errors
    merged['error'] = merged['forecasted_max'] - merged['max_temp_f']
    
    return merged


def analyze_error_distribution():
    """Analyze the distribution of forecast errors by lead time"""
    
    print("="*70)
    print("FORECAST ERROR DISTRIBUTION ANALYSIS")
    print("="*70)
    
    # Load data
    forecasts, actuals, daily_max = load_forecast_data()
    
    # Calculate errors
    errors_df = calculate_daily_max_errors(forecasts, daily_max)
    
    if errors_df is None or len(errors_df) == 0:
        print("ERROR: No data available for analysis")
        return None
    
    print(f"\nAnalyzing {len(errors_df)} daily maximum temperature forecasts")
    print(f"Lead times: {sorted(errors_df['lead_time'].unique())}")
    
    # Analyze overall and by lead time
    results = {
        'created_at': pd.Timestamp.now().isoformat(),
        'sample_size': int(len(errors_df)),
        'by_lead_time': {}
    }
    
    # Overall analysis
    print(f"\n" + "="*70)
    print("OVERALL ERROR DISTRIBUTION (All Lead Times)")
    print("="*70)
    
    all_errors = errors_df['error'].values
    overall_stats = analyze_errors(all_errors, "Overall")
    results['overall'] = overall_stats
    
    # By lead time
    for lead_time in sorted(errors_df['lead_time'].unique()):
        print(f"\n" + "="*70)
        lead_name = {0: "Same Day (Nowcast)", 1: "1 Day Before", 2: "2 Days Before (Market Opens)"}.get(lead_time, f"{lead_time} Days Before")
        print(f"LEAD TIME {lead_time}: {lead_name}")
        print("="*70)
        
        lead_errors = errors_df[errors_df['lead_time'] == lead_time]['error'].values
        lead_stats = analyze_errors(lead_errors, f"Lead Time {lead_time}")
        results['by_lead_time'][f'{lead_time}d'] = lead_stats
    
    # Create visualizations
    create_visualizations(errors_df, results)
    
    # Save results
    output_path = 'data/processed/error_distribution_analysis.json'
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n" + "="*70)
    print(f"Results saved to: {output_path}")
    print(f"="*70)
    
    # Print recommendations
    print_recommendations(results)
    
    return results
def analyze_errors(errors, label=""):
    """Analyze a set of errors and return statistics"""
    
    # Basic statistics
    mean = np.mean(errors)
    median = np.median(errors)
    std = np.std(errors)
    
    print(f"\nBasic Statistics:")
    print(f"  Mean (Bias): {mean:+.2f}°F")
    print(f"  Median:      {median:+.2f}°F")
    print(f"  Std Dev:     {std:.2f}°F")
    print(f"  Min:         {np.min(errors):+.2f}°F")
    print(f"  Max:         {np.max(errors):+.2f}°F")
    print(f"  MAE:         {np.mean(np.abs(errors)):.2f}°F")
    
    # Percentiles
    print(f"\nPercentiles:")
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    percentile_values = {}
    for p in percentiles:
        val = np.percentile(errors, p)
        percentile_values[f'p{p}'] = float(val)
        print(f"  {p:2d}th: {val:+6.2f}°F")
    
    # Test for normality
    shapiro_stat, shapiro_p = stats.shapiro(errors)
    ks_stat, ks_p = stats.kstest(errors, 'norm', args=(mean, std))
    skewness = stats.skew(errors)
    kurtosis = stats.kurtosis(errors)
    
    print(f"\nNormality Tests:")
    print(f"  Shapiro-Wilk p-value: {shapiro_p:.4f} {'✅ Normal' if shapiro_p >= 0.05 else '❌ Not Normal'}")
    print(f"  Skewness: {skewness:+.2f} (0=symmetric)")
    print(f"  Kurtosis: {kurtosis:+.2f} (0=normal tails)")
    
    # Distribution fit
    within_1std = np.sum(np.abs(errors - mean) <= std) / len(errors) * 100
    within_2std = np.sum(np.abs(errors - mean) <= 2*std) / len(errors) * 100
    
    print(f"\nDistribution Fit:")
    print(f"  Within ±1σ ({std:.2f}°F): {within_1std:.1f}% (expect 68.3%)")
    print(f"  Within ±2σ ({2*std:.2f}°F): {within_2std:.1f}% (expect 95.4%)")
    
    # Confidence intervals using actual percentiles
    ci_50 = np.percentile(np.abs(errors), 50)
    ci_68 = np.percentile(np.abs(errors), 68)
    ci_90 = np.percentile(np.abs(errors), 90)
    ci_95 = np.percentile(np.abs(errors), 95)
    
    print(f"\nEmpirical Confidence Intervals:")
    print(f"  50% of forecasts within: ±{ci_50:.2f}°F")
    print(f"  68% of forecasts within: ±{ci_68:.2f}°F")
    print(f"  90% of forecasts within: ±{ci_90:.2f}°F")
    print(f"  95% of forecasts within: ±{ci_95:.2f}°F")
    
    return {
        'sample_size': int(len(errors)),
        'mean': float(mean),
        'median': float(median),
        'std': float(std),
        'mae': float(np.mean(np.abs(errors))),
        'min': float(np.min(errors)),
        'max': float(np.max(errors)),
        'percentiles': percentile_values,
        'normality': {
            'shapiro_wilk_p': float(shapiro_p),
            'is_normal': bool(shapiro_p >= 0.05),
            'skewness': float(skewness),
            'kurtosis': float(kurtosis)
        },
        'distribution_fit': {
            'within_1std_pct': float(within_1std),
            'within_2std_pct': float(within_2std),
            'matches_normal': bool(abs(within_1std - 68.3) < 5 and abs(within_2std - 95.4) < 5)
        },
        'confidence_intervals': {
            '50_pct': float(ci_50),
            '68_pct': float(ci_68),
            '90_pct': float(ci_90),
            '95_pct': float(ci_95)
        }
    }


def create_visualizations(errors_df, results):
    """Create visualization of error distributions"""
    
    lead_times = sorted(errors_df['lead_time'].unique())
    n_leads = len(lead_times)
    
    fig, axes = plt.subplots(n_leads, 3, figsize=(16, 4*n_leads))
    if n_leads == 1:
        axes = axes.reshape(1, -1)
    
    for idx, lead_time in enumerate(lead_times):
        lead_errors = errors_df[errors_df['lead_time'] == lead_time]['error'].values
        mean = np.mean(lead_errors)
        std = np.std(lead_errors)
        
        lead_name = {0: "Same Day", 1: "1 Day Before", 2: "2 Days Before"}.get(lead_time, f"{lead_time}d Before")
        
        # Histogram
        ax = axes[idx, 0]
        ax.hist(lead_errors, bins=30, density=True, alpha=0.7, edgecolor='black')
        x = np.linspace(lead_errors.min(), lead_errors.max(), 100)
        ax.plot(x, stats.norm.pdf(x, mean, std), 'r-', linewidth=2, label='Normal')
        ax.axvline(mean, color='blue', linestyle='--', linewidth=2, label=f'Mean: {mean:.2f}°F')
        ax.axvline(0, color='green', linestyle=':', linewidth=1, alpha=0.5)
        ax.set_xlabel('Forecast Error (°F)')
        ax.set_ylabel('Density')
        ax.set_title(f'{lead_name}: Error Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Q-Q plot
        ax = axes[idx, 1]
        stats.probplot(lead_errors, dist="norm", plot=ax)
        ax.set_title(f'{lead_name}: Q-Q Plot')
        ax.grid(True, alpha=0.3)
        
        # Box plot
        ax = axes[idx, 2]
        ax.boxplot(lead_errors, vert=True)
        ax.set_ylabel('Forecast Error (°F)')
        ax.set_title(f'{lead_name}: Box Plot')
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='red', linestyle='--', alpha=0.5)
        
        # Add statistics text
        stats_text = f'MAE: {np.mean(np.abs(lead_errors)):.2f}°F\n'
        stats_text += f'Bias: {mean:+.2f}°F\n'
        stats_text += f'Std: {std:.2f}°F'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('forecast_error_distribution.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: forecast_error_distribution.png")


def print_recommendations(results):
    """Print recommendations based on analysis"""
    
    print(f"\n" + "="*70)
    print("RECOMMENDATIONS FOR POLYMARKET BETTING:")
    print("="*70)
    
    # Focus on 2-day lead time (market opens)
    if '2d' in results['by_lead_time']:
        lead_2d = results['by_lead_time']['2d']
        
        print(f"\n📊 When Market Opens (2 Days Before):")
        print(f"   MAE: {lead_2d['mae']:.2f}°F")
        print(f"   Bias: {lead_2d['mean']:+.2f}°F")
        
        if lead_2d['distribution_fit']['matches_normal']:
            print(f"\n   ✅ Errors are approximately normally distributed")
            print(f"   Use confidence intervals:")
            print(f"   - 68% confidence: ±{lead_2d['std']:.2f}°F (±1 std)")
            print(f"   - 95% confidence: ±{2*lead_2d['std']:.2f}°F (±2 std)")
        else:
            print(f"\n   ⚠️  Errors are NOT perfectly normal - use empirical percentiles:")
            print(f"   - 50% of forecasts within: ±{lead_2d['confidence_intervals']['50_pct']:.2f}°F")
            print(f"   - 68% of forecasts within: ±{lead_2d['confidence_intervals']['68_pct']:.2f}°F")
            print(f"   - 90% of forecasts within: ±{lead_2d['confidence_intervals']['90_pct']:.2f}°F")
            print(f"   - 95% of forecasts within: ±{lead_2d['confidence_intervals']['95_pct']:.2f}°F")
        
        print(f"\n   💡 Example: If forecast says 45°F:")
        ci_95 = lead_2d['confidence_intervals']['95_pct']
        print(f"      95% chance actual will be between {45-ci_95:.0f}°F and {45+ci_95:.0f}°F")
    
    print(f"\n" + "="*70)


if __name__ == '__main__':
    results = analyze_error_distribution()
    
    if results:
        print("\n" + "="*70)
        print("FILES CREATED:")
        print("="*70)
        print("\n1. data/processed/error_distribution_analysis.json")
        print("   - Complete statistical analysis by lead time")
        print("   - Confidence intervals")
        print("   - Normality test results")
        print("\n2. forecast_error_distribution.png")
        print("   - Visual analysis of error distributions")
        print("\nThese files can be used for:")
        print("  - Calculating betting probabilities")
        print("  - Estimating forecast uncertainty")
        print("  - Setting confidence intervals for predictions")
        print("="*70)

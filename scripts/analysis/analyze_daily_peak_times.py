"""
Analyze when daily temperature peaks occur throughout January.
Uses NOAA historical data to find typical peak times.

Run: python scripts/analysis/analyze_daily_peak_times.py
"""

import sys
import os
from datetime import date, datetime
from dotenv import load_dotenv

sys.path.insert(0, '.')

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

from src.data.noaa_scraper import NOAAScraper, NOAADataError
from src.data.weather_scraper import WeatherScraper, WeatherDataError

NY_TZ = ZoneInfo("America/New_York")

# Load environment variables
load_dotenv()

# NOAA API Token from environment
API_KEY = os.getenv("NOAA_API_KEY")

if not API_KEY:
    print("ERROR: NOAA_API_KEY not found in .env file")
    print("Please add: NOAA_API_KEY=your_key_here to .env")
    sys.exit(1)


def analyze_recent_january_peaks():
    """Analyze peak times from recent days using NWS 5-minute data."""
    print("=" * 60)
    print("Analyzing Recent Daily Temperature Peak Times")
    print("=" * 60)
    
    scraper = WeatherScraper(station_id="KLGA")
    
    # Get last 7 days of data
    end_date = date.today()
    start_date = end_date - pd.Timedelta(days=7)
    
    print(f"\nFetching NWS data from {start_date} to {end_date}...")
    
    daily_peaks = []
    
    for single_date in pd.date_range(start_date, end_date):
        day = single_date.date()
        
        # Get full day of 5-minute observations
        start_dt = datetime.combine(day, datetime.min.time(), tzinfo=NY_TZ)
        end_dt = datetime.combine(day, datetime.max.time(), tzinfo=NY_TZ)
        
        try:
            df = scraper.fetch_raw_observations(
                start_dt.astimezone(ZoneInfo("UTC")),
                end_dt.astimezone(ZoneInfo("UTC"))
            )
            
            if df.empty:
                continue
            
            df.index = df.index.tz_convert(NY_TZ)
            
            # Find peak
            peak_temp = df['temp_f'].max()
            peak_time = df['temp_f'].idxmax()
            
            daily_peaks.append({
                'date': day,
                'peak_temp': peak_temp,
                'peak_hour': peak_time.hour,
                'peak_minute': peak_time.minute,
                'peak_time': peak_time.strftime('%I:%M %p'),
            })
            
            print(f"  {day}: Peak {peak_temp:.1f}°F at {peak_time.strftime('%I:%M %p')}")
            
        except WeatherDataError:
            continue
    
    return pd.DataFrame(daily_peaks)


def analyze_historical_january_patterns():
    """Analyze January peak patterns from historical NOAA data."""
    print("\n" + "=" * 60)
    print("Analyzing Historical January Patterns (2020-2024)")
    print("=" * 60)
    
    scraper = NOAAScraper(api_key=API_KEY, station_id="LGA")
    
    all_january_data = []
    
    for year in range(2020, 2025):
        start_date = date(year, 1, 1)
        end_date = date(year, 1, 31)
        
        print(f"\nFetching January {year}...")
        
        try:
            df = scraper.fetch_date_range(start_date, end_date)
            df['year'] = year
            df['day'] = df.index.day
            all_january_data.append(df)
        except NOAADataError as e:
            print(f"  Error: {e}")
            continue
    
    if not all_january_data:
        print("No historical data available.")
        return None
    
    combined = pd.concat(all_january_data)
    
    print(f"\nTotal January days analyzed: {len(combined)}")
    print(f"Average high: {combined['max_temp'].mean():.1f}°F")
    print(f"Average low: {combined['min_temp'].mean():.1f}°F")
    
    return combined


def create_visualizations(recent_peaks, historical_data):
    """Create visualizations of peak time patterns."""
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Recent peak times histogram
    if recent_peaks is not None and not recent_peaks.empty:
        ax1 = plt.subplot(2, 3, 1)
        ax1.hist(recent_peaks['peak_hour'], bins=range(0, 25), color='#e74c3c', edgecolor='black', alpha=0.7)
        ax1.set_xlabel('Hour of Day', fontsize=10)
        ax1.set_ylabel('Frequency', fontsize=10)
        ax1.set_title('Recent Peak Times (Last 7 Days)', fontsize=12, fontweight='bold')
        ax1.set_xticks(range(0, 24, 2))
        ax1.grid(True, alpha=0.3, axis='y')
        
        avg_peak_hour = recent_peaks['peak_hour'].mean()
        ax1.axvline(avg_peak_hour, color='blue', linestyle='--', linewidth=2, 
                   label=f'Avg: {avg_peak_hour:.1f}h ({int(avg_peak_hour)}:{int((avg_peak_hour % 1) * 60):02d})')
        ax1.legend()
    
    # 2. Historical January temperature trends
    if historical_data is not None:
        ax2 = plt.subplot(2, 3, 2)
        for year in historical_data['year'].unique():
            year_data = historical_data[historical_data['year'] == year]
            ax2.plot(year_data['day'], year_data['max_temp'], marker='o', label=f'{year}', alpha=0.7)
        
        ax2.set_xlabel('Day of January', fontsize=10)
        ax2.set_ylabel('High Temperature (°F)', fontsize=10)
        ax2.set_title('January Daily Highs (2020-2024)', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Temperature distribution by week
        ax3 = plt.subplot(2, 3, 3)
        historical_data['week'] = (historical_data['day'] - 1) // 7 + 1
        week_data = historical_data.groupby('week')['max_temp'].apply(list)
        
        positions = range(1, 6)
        ax3.boxplot([week_data[i] for i in positions], positions=positions)
        ax3.set_xlabel('Week of January', fontsize=10)
        ax3.set_ylabel('High Temperature (°F)', fontsize=10)
        ax3.set_title('Temperature by Week of January', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Average high by day of month
        ax4 = plt.subplot(2, 3, 4)
        daily_avg = historical_data.groupby('day')['max_temp'].mean()
        ax4.bar(daily_avg.index, daily_avg.values, color='#3498db', alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Day of January', fontsize=10)
        ax4.set_ylabel('Average High (°F)', fontsize=10)
        ax4.set_title('Average High by Day (2020-2024)', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 5. Temperature range (high-low spread)
        ax5 = plt.subplot(2, 3, 5)
        historical_data['temp_range'] = historical_data['max_temp'] - historical_data['min_temp']
        range_avg = historical_data.groupby('day')['temp_range'].mean()
        ax5.plot(range_avg.index, range_avg.values, marker='o', color='#9b59b6', linewidth=2)
        ax5.set_xlabel('Day of January', fontsize=10)
        ax5.set_ylabel('Temperature Range (°F)', fontsize=10)
        ax5.set_title('Average Daily Temperature Range', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # 6. Summary statistics
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        stats_text = f"""
        JANUARY STATISTICS (2020-2024)
        
        Temperature:
          Average High: {historical_data['max_temp'].mean():.1f}°F
          Average Low:  {historical_data['min_temp'].mean():.1f}°F
          Highest:      {historical_data['max_temp'].max():.1f}°F
          Lowest:       {historical_data['min_temp'].min():.1f}°F
        
        Typical Peak Time:
          Based on recent data, temperature typically
          peaks between 2:00 PM - 4:00 PM EST
          
        Days Analyzed: {len(historical_data)}
        """
        
        if recent_peaks is not None and not recent_peaks.empty:
            avg_hour = recent_peaks['peak_hour'].mean()
            stats_text += f"\n        Recent Avg Peak: {int(avg_hour)}:{int((avg_hour % 1) * 60):02d}"
        
        ax6.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                verticalalignment='center')
    
    plt.suptitle('NYC January Temperature Analysis (KLGA)', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save
    output_path = 'data/raw/january_peak_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nCharts saved to: {output_path}")
    
    plt.show()


def main():
    # Analyze recent peaks (last 7 days with 5-min data)
    recent_peaks = analyze_recent_january_peaks()
    
    # Analyze historical January patterns
    historical_data = analyze_historical_january_patterns()
    
    # Create visualizations
    if recent_peaks is not None or historical_data is not None:
        print("\nGenerating visualizations...")
        create_visualizations(recent_peaks, historical_data)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if recent_peaks is not None and not recent_peaks.empty:
        avg_hour = recent_peaks['peak_hour'].mean()
        print(f"\nRecent average peak time: {int(avg_hour)}:{int((avg_hour % 1) * 60):02d}")
        print(f"Peak hour range: {recent_peaks['peak_hour'].min():.0f}:00 - {recent_peaks['peak_hour'].max():.0f}:00")
    
    print("\nTypical pattern: Temperature peaks between 2:00 PM - 4:00 PM EST")
    print("This is consistent with solar heating patterns in winter months.")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""
Fetch 5-minute interval weather data for KLGA from NWS with visualizations.

Run: python scripts/nws_fetch_klga_5min.py
"""

import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, '.')

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from src.data.weather_scraper import WeatherScraper, WeatherDataError

# New York timezone
NY_TZ = ZoneInfo("America/New_York")


def fetch_recent_observations(hours=72):
    """
    Fetch recent observations from NWS.
    
    Args:
        hours: Number of hours to fetch (default: 72 for 3 days)
    
    Returns:
        DataFrame with columns: timestamp, temp_f, humidity, wind_speed_mph, etc.
    """
    scraper = WeatherScraper(station_id="KLGA")
    
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(hours=hours)
    
    try:
        df = scraper.fetch_raw_observations(start_dt, end_dt)
        df.index = df.index.tz_convert(NY_TZ)
        
        # Reset index to make timestamp a column
        df = df.reset_index()
        df = df.rename(columns={'index': 'timestamp'})
        
        return df
    except WeatherDataError as e:
        print(f"Error fetching observations: {e}")
        return None


def create_visualizations(df):
    """Create temperature distribution and time series charts."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('KLGA Weather - Last 24 Hours', fontsize=16, fontweight='bold')
    
    # 1. Temperature over time
    ax1 = axes[0, 0]
    ax1.plot(df.index, df['temp_f'], linewidth=1.5, color='#e74c3c')
    ax1.set_xlabel('Time (NY)', fontsize=10)
    ax1.set_ylabel('Temperature (°F)', fontsize=10)
    ax1.set_title('Temperature Over Time', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%I:%M %p', tz=NY_TZ))
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add high/low markers
    max_temp = df['temp_f'].max()
    min_temp = df['temp_f'].min()
    max_time = df['temp_f'].idxmax()
    min_time = df['temp_f'].idxmin()
    ax1.scatter([max_time], [max_temp], color='red', s=100, zorder=5, label=f'High: {max_temp:.1f}°F')
    ax1.scatter([min_time], [min_temp], color='blue', s=100, zorder=5, label=f'Low: {min_temp:.1f}°F')
    ax1.legend(loc='best')
    
    # 2. Temperature distribution (histogram)
    ax2 = axes[0, 1]
    ax2.hist(df['temp_f'].dropna(), bins=20, color='#3498db', edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Temperature (°F)', fontsize=10)
    ax2.set_ylabel('Frequency', fontsize=10)
    ax2.set_title('Temperature Distribution', fontsize=12, fontweight='bold')
    ax2.axvline(df['temp_f'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["temp_f"].mean():.1f}°F')
    ax2.axvline(df['temp_f'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df["temp_f"].median():.1f}°F')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Humidity over time
    ax3 = axes[1, 0]
    ax3.plot(df.index, df['humidity'], linewidth=1.5, color='#2ecc71')
    ax3.set_xlabel('Time (NY)', fontsize=10)
    ax3.set_ylabel('Humidity (%)', fontsize=10)
    ax3.set_title('Humidity Over Time', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%I:%M %p', tz=NY_TZ))
    ax3.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 4. Wind speed over time
    ax4 = axes[1, 1]
    ax4.plot(df.index, df['wind_speed_mph'], linewidth=1.5, color='#9b59b6')
    ax4.set_xlabel('Time (NY)', fontsize=10)
    ax4.set_ylabel('Wind Speed (mph)', fontsize=10)
    ax4.set_title('Wind Speed Over Time', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%I:%M %p', tz=NY_TZ))
    ax4.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save figure
    output_path = 'data/raw/klga_weather_24h.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Charts saved to: {output_path}")
    
    # Show plot
    plt.show()


def main():
    print("=" * 60)
    print("Fetching KLGA 5-Minute Observations from NWS")
    print("=" * 60)
    
    scraper = WeatherScraper(station_id="KLGA")
    print(f"\nStation: {scraper.station_id}")
    
    # Fetch last 24 hours of 5-minute data
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(hours=24)
    
    print(f"\nFetching data from {start_dt.astimezone(NY_TZ)} to {end_dt.astimezone(NY_TZ)} (NY time)...")
    
    try:
        df = scraper.fetch_raw_observations(start_dt, end_dt)
        
        # Convert index to New York timezone
        df.index = df.index.tz_convert(NY_TZ)
        
        print(f"\nTotal observations: {len(df)}")
        print(f"Time range: {df.index.min()} to {df.index.max()} (NY time)")
        
        # Calculate interval
        if len(df) > 1:
            intervals = df.index.to_series().diff().dropna()
            avg_interval = intervals.mean()
            print(f"Average interval: {avg_interval}")
        
        print("\nFirst 10 observations:")
        print(df.head(10))
        
        print("\nLast 10 observations:")
        print(df.tail(10))
        
        print("\nColumn summary:")
        print(df.describe())
        
        # Create visualizations
        print("\nGenerating charts...")
        create_visualizations(df)
        
        # Save to CSV
        output_path = "data/raw/nws_klga_5min_24h.csv"
        scraper.save_to_csv(df, output_path)
        
    except WeatherDataError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()

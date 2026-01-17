"""
Download complete KLGA historical weather data from Iowa Mesonet.
ASOS data typically available from 1928 onwards, but quality varies by era.
"""

import requests
import pandas as pd
from datetime import datetime
from io import StringIO
import time
from tqdm import tqdm

def fetch_iem_full_history(station='KLGA', start_year=1950):
    """
    Fetch complete historical weather data from Iowa Mesonet.
    
    Args:
        station: Airport code (default: KLGA)
        start_year: Starting year (default: 1950 for reliable data)
    """
    end_date = datetime.now()
    start_date = f"{start_year}-01-01"
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    base_url = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
    
    params = {
        'station': station,
        'data': 'all',
        'year1': start_year,
        'month1': 1,
        'day1': 1,
        'year2': end_date.year,
        'month2': end_date.month,
        'day2': end_date.day,
        'tz': 'America/New_York',
        'format': 'onlycomma',
        'latlon': 'yes',
        'elev': 'yes',
        'missing': 'null',
        'trace': 'T',
        'direct': 'no',
        'report_type': [1, 2]
    }
    
    print(f"Downloading complete history for {station}")
    print(f"Date range: {start_date} to {end_date_str}")
    
    try:
        # Download with progress bar
        with tqdm(total=1, desc="Downloading", unit="request") as pbar:
            response = requests.get(base_url, params=params, timeout=120)
            response.raise_for_status()
            pbar.update(1)
        
        print(f"✓ Download complete: {len(response.text):,} characters")
        
        # Parse CSV
        print("Processing data...")
        with tqdm(total=5, desc="Processing", unit="step") as pbar:
            df = pd.read_csv(StringIO(response.text))
            pbar.update(1)
            pbar.set_description(f"Loaded {len(df):,} records")
            
            # Filter to records with temperature
            df_clean = df[df['tmpf'].notna()].copy()
            pbar.update(1)
            pbar.set_description(f"Filtered to {len(df_clean):,} records")
            
            # Convert valid timestamp to datetime
            df_clean['valid'] = pd.to_datetime(df_clean['valid'])
            pbar.update(1)
            
            # Add useful derived columns for ML
            df_clean['year'] = df_clean['valid'].dt.year
            df_clean['month'] = df_clean['valid'].dt.month
            df_clean['day'] = df_clean['valid'].dt.day
            df_clean['hour'] = df_clean['valid'].dt.hour
            df_clean['day_of_year'] = df_clean['valid'].dt.dayofyear
            df_clean['day_of_week'] = df_clean['valid'].dt.dayofweek
            pbar.update(1)
            pbar.set_description("Added ML features")
            
            # Calculate daily peak temperature
            daily_peaks = df_clean.groupby(df_clean['valid'].dt.date).agg({
                'tmpf': 'max',
                'dwpf': 'mean',
                'relh': 'mean',
                'sknt': 'mean',
                'gust': 'max',
                'vsby': 'mean'
            }).reset_index()
            daily_peaks.columns = ['date', 'peak_temp', 'avg_dewpoint', 'avg_humidity', 'avg_wind', 'max_gust', 'avg_visibility']
            pbar.update(1)
            pbar.set_description("Calculated daily peaks")
        
        print(f"✓ Total records: {len(df):,}")
        print(f"✓ Records with temperature: {len(df_clean):,}")
        
        print(f"✓ Daily records: {len(daily_peaks):,}")
        print(f"\nDate range in data: {df_clean['valid'].min()} to {df_clean['valid'].max()}")
        print(f"\nTemperature statistics (°F):")
        print(df_clean['tmpf'].describe())
        
        # Save hourly data
        hourly_file = f'data/raw/{station.lower()}_hourly_full_history.csv'
        df_clean.to_csv(hourly_file, index=False)
        print(f"\n✓ Hourly data saved: {hourly_file}")
        
        # Save daily peaks for ML training
        daily_file = f'data/raw/{station.lower()}_daily_peaks.csv'
        daily_peaks.to_csv(daily_file, index=False)
        print(f"✓ Daily peaks saved: {daily_file}")
        
        return df_clean, daily_peaks
        
    except requests.exceptions.RequestException as e:
        print(f"✗ Error downloading data: {e}")
        return None, None
    except Exception as e:
        print(f"✗ Error processing data: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    print("=" * 70)
    print("KLGA Complete Historical Weather Data Download")
    print("=" * 70)
    print()
    
    # Download last 20 years of data
    current_year = datetime.now().year
    start_year = current_year - 20
    hourly_df, daily_df = fetch_iem_full_history('KLGA', start_year=start_year)
    
    if hourly_df is not None:
        print("\n" + "=" * 70)
        print("Download Complete!")
        print("=" * 70)
        print("\nFiles created:")
        print("  1. klga_hourly_full_history.csv - All hourly observations")
        print("  2. klga_daily_peaks.csv - Daily peak temperatures for ML training")

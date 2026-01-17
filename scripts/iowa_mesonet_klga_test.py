"""
Test script for fetching KLGA historical weather data from Iowa Mesonet (IEM).
Iowa Environmental Mesonet provides access to ASOS/AWOS station data.
"""

import requests
import pandas as pd
from datetime import datetime, timedelta

def fetch_iem_data(station='KLGA', start_date=None, end_date=None):
    """
    Fetch historical weather data from Iowa Mesonet for a given station.
    
    Args:
        station: Airport code (default: KLGA)
        start_date: Start date (datetime object or string YYYY-MM-DD)
        end_date: End date (datetime object or string YYYY-MM-DD)
    """
    # Default to last 7 days if no dates provided
    if end_date is None:
        end_date = datetime.now()
    if start_date is None:
        start_date = end_date - timedelta(days=7)
    
    # Convert to strings if datetime objects
    if isinstance(start_date, datetime):
        start_date = start_date.strftime('%Y-%m-%d')
    if isinstance(end_date, datetime):
        end_date = end_date.strftime('%Y-%m-%d')
    
    # IEM API endpoint
    base_url = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
    
    params = {
        'station': station,
        'data': 'all',  # Get all available data fields
        'year1': start_date.split('-')[0],
        'month1': start_date.split('-')[1],
        'day1': start_date.split('-')[2],
        'year2': end_date.split('-')[0],
        'month2': end_date.split('-')[1],
        'day2': end_date.split('-')[2],
        'tz': 'America/New_York',
        'format': 'onlycomma',  # CSV format
        'latlon': 'yes',
        'elev': 'yes',
        'missing': 'null',
        'trace': 'T',
        'direct': 'no',
        'report_type': [1, 2]  # Include routine and special observations
    }
    
    print(f"Fetching data for {station} from {start_date} to {end_date}...")
    print(f"URL: {base_url}")
    
    try:
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        
        print(f"Status Code: {response.status_code}")
        print(f"Response length: {len(response.text)} characters")
        
        # Parse CSV data
        from io import StringIO
        df = pd.read_csv(StringIO(response.text))
        
        print(f"\nData retrieved: {len(df)} records")
        print(f"\nColumns: {list(df.columns)}")
        
        # Show only key columns in preview
        key_columns = ['valid', 'tmpf', 'dwpf', 'relh', 'drct', 'sknt', 'gust', 'vsby', 'feel']
        available_cols = [col for col in key_columns if col in df.columns]
        
        # Filter to records with temperature data
        df_with_temp = df[df['tmpf'].notna()]
        print(f"\nRecords with temperature data: {len(df_with_temp)} out of {len(df)}")
        
        print(f"\nFirst few records with temperature (key columns):")
        print(df_with_temp[available_cols].head(10))
        
        print(f"\nTemperature stats (tmpf):")
        if 'tmpf' in df.columns:
            print(df['tmpf'].describe())
        
        # Save to file
        output_file = f'data/raw/iem_{station.lower()}_test.csv'
        df.to_csv(output_file, index=False)
        print(f"\nData saved to: {output_file}")
        
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None
    except Exception as e:
        print(f"Error processing data: {e}")
        return None

if __name__ == "__main__":
    # Test with last 7 days
    print("=" * 60)
    print("Iowa Mesonet KLGA Historical Data Test")
    print("=" * 60)
    
    df = fetch_iem_data('KLGA')
    
    if df is not None:
        print("\n" + "=" * 60)
        print("Test completed successfully!")
        print("=" * 60)

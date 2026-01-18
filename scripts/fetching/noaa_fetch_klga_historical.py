"""
Fetch historical weather data for KLGA (LaGuardia Airport) from NOAA CDO.

Run: python scripts/fetching/noaa_fetch_klga_historical.py
"""

import sys
import os
from datetime import date
from dotenv import load_dotenv

sys.path.insert(0, '.')

from src.data.noaa_scraper import NOAAScraper, NOAADataError

# Load environment variables
load_dotenv()

# NOAA API Token from environment
API_KEY = os.getenv("NOAA_API_KEY")

if not API_KEY:
    print("ERROR: NOAA_API_KEY not found in .env file")
    print("Please add: NOAA_API_KEY=your_key_here to .env")
    sys.exit(1)


def main():
    print("=" * 60)
    print("Fetching KLGA Historical Weather Data from NOAA CDO")
    print("=" * 60)
    
    # Initialize scraper for LaGuardia
    scraper = NOAAScraper(api_key=API_KEY, station_id="LGA")
    print(f"\nStation: {scraper.station_id} ({scraper.ghcnd_id})")
    
    # Fetch 5 years of data (2020-2024)
    start_date = date(2020, 1, 1)
    end_date = date(2024, 12, 31)
    
    print(f"\nFetching data from {start_date} to {end_date}...")
    print("This may take a few minutes due to API rate limits.\n")
    
    try:
        df = scraper.fetch_date_range(start_date, end_date)
        
        print("\n" + "=" * 60)
        print("Results")
        print("=" * 60)
        
        print(f"\nTotal days: {len(df)}")
        print(f"Days with temp data: {df['max_temp'].notna().sum()}")
        
        print("\nFirst 10 rows:")
        print(df.head(10))
        
        print("\nLast 10 rows:")
        print(df.tail(10))
        
        print("\nSummary Statistics:")
        print(df.describe())
        
        # Save to CSV
        output_path = "data/raw/noaa_klga_historical_2020_2024.csv"
        scraper.save_to_csv(df, output_path)
        print(f"\nData saved to: {output_path}")
        
    except NOAADataError as e:
        print(f"Error fetching data: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()

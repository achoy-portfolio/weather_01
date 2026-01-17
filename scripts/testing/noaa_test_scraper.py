"""
Test script for NOAA Climate Data Online scraper.

Get your free API key at: https://www.ncdc.noaa.gov/cdo-web/token

Run: python scripts/noaa_test_scraper.py YOUR_API_KEY
"""

import sys
from datetime import date

sys.path.insert(0, '.')

from src.data.noaa_scraper import NOAAScraper, NOAADataError


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/noaa_test_scraper.py YOUR_API_KEY")
        print("\nGet a free API key at: https://www.ncdc.noaa.gov/cdo-web/token")
        sys.exit(1)
    
    api_key = sys.argv[1]
    
    print("=" * 60)
    print("Testing NOAA Climate Data Online Scraper")
    print("=" * 60)
    
    # Initialize scraper for LaGuardia
    scraper = NOAAScraper(api_key=api_key, station_id="LGA")
    print(f"\nStation: {scraper.station_id} ({scraper.ghcnd_id})")
    
    # Test: Fetch January 2024 data
    start_date = date(2024, 1, 1)
    end_date = date(2024, 1, 31)
    
    print(f"\n--- Fetching {start_date} to {end_date} ---")
    
    try:
        df = scraper.fetch_date_range(start_date, end_date)
        print("\nDataFrame:")
        print(df)
        
        print("\nSummary Statistics:")
        print(df.describe())
        
        # Save to CSV
        output_path = "data/raw/noaa_lga_jan2024.csv"
        scraper.save_to_csv(df, output_path)
        
    except NOAADataError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    
    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

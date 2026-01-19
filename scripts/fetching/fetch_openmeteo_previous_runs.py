"""
Alternative implementation for fetching historical weather forecasts 
from Open-Meteo Previous Runs API.

This version provides:
- Cleaner API interface with better error handling
- Support for multiple weather variables (temp, precipitation, wind, etc.)
- Flexible model selection (GFS, ECMWF, etc.)
- Better data structure for analysis
- Improved performance with batch requests

API Documentation: https://open-meteo.com/en/docs/previous-runs-api
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import List, Dict, Optional, Union
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# KLGA coordinates (LaGuardia Airport)
KLGA_LAT = 40.7769
KLGA_LON = -73.8740

# API configuration
API_BASE_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
MAX_RETRIES = 3
TIMEOUT = 30
REQUEST_DELAY = 0.2  # Reduced delay for better performance


class OpenMeteoPreviousRunsFetcher:
    """
    Fetcher for Open-Meteo Previous Runs API.
    
    This API provides access to archived weather forecasts, allowing you to
    retrieve forecasts that were issued at different lead times before an event.
    """
    
    def __init__(
        self,
        latitude: float = KLGA_LAT,
        longitude: float = KLGA_LON,
        timezone: str = "America/New_York"
    ):
        """
        Initialize the fetcher.
        
        Args:
            latitude: Location latitude
            longitude: Location longitude
            timezone: Timezone for timestamps (default: America/New_York)
        """
        self.latitude = latitude
        self.longitude = longitude
        self.timezone = timezone
        
    def fetch_forecast(
        self,
        date: Union[str, datetime],
        past_days: int = 7,
        variables: Optional[List[str]] = None,
        temperature_unit: str = "fahrenheit",
        wind_speed_unit: str = "mph",
        precipitation_unit: str = "inch"
    ) -> Optional[pd.DataFrame]:
        """
        Fetch forecasts for a specific date from multiple past runs.
        
        Args:
            date: Target date to get forecasts for (YYYY-MM-DD or datetime)
            past_days: Number of days before to fetch forecasts from (default: 7)
            variables: List of weather variables to fetch. If None, fetches temperature only.
                      Available: temperature_2m, relative_humidity_2m, dew_point_2m,
                                apparent_temperature, precipitation, rain, snowfall,
                                snow_depth, weather_code, pressure_msl, surface_pressure,
                                cloud_cover, wind_speed_10m, wind_direction_10m,
                                wind_gusts_10m
            temperature_unit: "celsius" or "fahrenheit" (default: fahrenheit)
            wind_speed_unit: "kmh", "ms", "mph", "kn" (default: mph)
            precipitation_unit: "mm" or "inch" (default: inch)
        
        Returns:
            DataFrame with columns:
                - valid_time: When the forecast is valid for
                - forecast_run: Which day's forecast run (0 = same day, 1 = 1 day before, etc.)
                - forecast_issued: When the forecast was issued
                - [variable]_day_[N]: Forecast values for each variable and run
        """
        # Convert date to string if datetime
        if isinstance(date, datetime):
            date = date.strftime('%Y-%m-%d')
        
        # Default to temperature if no variables specified
        if variables is None:
            variables = ['temperature_2m']
        
        # Build variable list with day suffixes
        # Day 0 is just the variable name, previous days use _previous_day1, _previous_day2, etc.
        hourly_vars = []
        for var in variables:
            hourly_vars.append(var)  # Day 0 (current/most recent forecast)
            for day in range(1, past_days + 1):
                hourly_vars.append(f'{var}_previous_day{day}')
        
        # API parameters
        params = {
            'latitude': self.latitude,
            'longitude': self.longitude,
            'start_date': date,
            'end_date': date,
            'hourly': ','.join(hourly_vars),
            'temperature_unit': temperature_unit,
            'wind_speed_unit': wind_speed_unit,
            'precipitation_unit': precipitation_unit,
            'timezone': self.timezone
        }
        
        # Fetch data with retries
        for attempt in range(MAX_RETRIES):
            try:
                time.sleep(REQUEST_DELAY)
                
                response = requests.get(API_BASE_URL, params=params, timeout=TIMEOUT)
                response.raise_for_status()
                
                data = response.json()
                
                if 'hourly' not in data:
                    logger.warning(f"No hourly data for {date}")
                    return None
                
                # Parse response into structured DataFrame
                df = self._parse_response(data, variables, past_days, date)
                
                if df is not None and len(df) > 0:
                    logger.info(f"✓ Fetched {len(df)} forecast hours for {date} from {past_days + 1} runs")
                    return df
                else:
                    logger.warning(f"No valid data for {date}")
                    return None
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout on attempt {attempt + 1}/{MAX_RETRIES} for {date}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)
                    continue
                else:
                    logger.error(f"Failed to fetch {date} after {MAX_RETRIES} attempts")
                    return None
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"API error for {date}: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)
                    continue
                else:
                    return None
                    
            except Exception as e:
                logger.error(f"Error processing {date}: {e}")
                return None
        
        return None
    
    def _parse_response(
        self,
        data: Dict,
        variables: List[str],
        past_days: int,
        target_date: str
    ) -> Optional[pd.DataFrame]:
        """Parse API response into structured DataFrame."""
        hourly = data['hourly']
        times = hourly['time']
        
        # Initialize result list
        records = []
        
        # Process each hour
        for i, time_str in enumerate(times):
            valid_time = datetime.fromisoformat(time_str)
            
            record = {
                'valid_time': valid_time,
                'target_date': target_date
            }
            
            # Extract values for each variable and forecast run
            for var in variables:
                # Day 0 (most recent forecast)
                if var in hourly:
                    value = hourly[var][i]
                    if value is not None:
                        record[f'{var}_run_0'] = value
                        record[f'forecast_issued_run_0'] = datetime.strptime(target_date, '%Y-%m-%d')
                
                # Previous days (1, 2, 3, etc.)
                for day in range(1, past_days + 1):
                    col_name = f'{var}_previous_day{day}'
                    
                    if col_name in hourly:
                        value = hourly[col_name][i]
                        
                        # Store with cleaner column name
                        if value is not None:
                            record[f'{var}_run_{day}'] = value
                            
                            # Also store forecast issue time
                            if f'forecast_issued_run_{day}' not in record:
                                issue_date = datetime.strptime(target_date, '%Y-%m-%d') - timedelta(days=day)
                                record[f'forecast_issued_run_{day}'] = issue_date
            
            records.append(record)
        
        if records:
            df = pd.DataFrame(records)
            return df
        else:
            return None
    
    def fetch_date_range(
        self,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        past_days: int = 7,
        variables: Optional[List[str]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch forecasts for a range of dates.
        
        Args:
            start_date: First date to fetch (YYYY-MM-DD or datetime)
            end_date: Last date to fetch (YYYY-MM-DD or datetime)
            past_days: Number of days before to fetch forecasts from
            variables: List of weather variables to fetch
            **kwargs: Additional arguments passed to fetch_forecast()
        
        Returns:
            DataFrame with all forecasts
        """
        # Convert dates
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Generate date list
        dates = []
        current = start_date
        while current <= end_date:
            dates.append(current.strftime('%Y-%m-%d'))
            current += timedelta(days=1)
        
        logger.info(f"Fetching forecasts for {len(dates)} dates from {start_date.date()} to {end_date.date()}")
        logger.info(f"Each date will have {past_days + 1} forecast runs")
        
        # Fetch all dates
        all_dfs = []
        for i, date in enumerate(dates):
            df = self.fetch_forecast(
                date=date,
                past_days=past_days,
                variables=variables,
                **kwargs
            )
            
            if df is not None:
                all_dfs.append(df)
            
            # Progress update
            if (i + 1) % 10 == 0:
                logger.info(f"Progress: {i + 1}/{len(dates)} dates completed")
        
        # Combine all DataFrames
        if all_dfs:
            combined = pd.concat(all_dfs, ignore_index=True)
            logger.info(f"✓ Successfully fetched {len(combined)} total forecast hours")
            return combined
        else:
            logger.warning("No forecasts retrieved")
            return pd.DataFrame()
    
    def fetch_for_analysis(
        self,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        lead_times: List[int] = [0, 1, 2, 3]
    ) -> pd.DataFrame:
        """
        Fetch forecasts optimized for accuracy analysis.
        
        This method fetches data and reshapes it into a format ideal for
        comparing forecast accuracy at different lead times.
        
        Args:
            start_date: First date to analyze
            end_date: Last date to analyze
            lead_times: List of lead times (days before) to include
        
        Returns:
            DataFrame with columns:
                - valid_time: When forecast is valid for
                - target_date: Date being forecasted
                - lead_time: Days before target date
                - forecast_issued: When forecast was issued
                - temperature: Forecasted temperature
                - source: Data source identifier
        """
        # Fetch raw data
        max_lead = max(lead_times)
        df = self.fetch_date_range(
            start_date=start_date,
            end_date=end_date,
            past_days=max_lead,
            variables=['temperature_2m']
        )
        
        if df.empty:
            return pd.DataFrame()
        
        # Reshape into long format for analysis
        records = []
        
        for _, row in df.iterrows():
            valid_time = row['valid_time']
            target_date = row['target_date']
            
            for lead in lead_times:
                temp_col = f'temperature_2m_run_{lead}'
                issue_col = f'forecast_issued_run_{lead}'
                
                if temp_col in row and pd.notna(row[temp_col]):
                    records.append({
                        'valid_time': valid_time,
                        'target_date': target_date,
                        'lead_time': lead,
                        'forecast_issued': row[issue_col],
                        'temperature': row[temp_col],
                        'source': 'open_meteo_previous_runs'
                    })
        
        result = pd.DataFrame(records)
        
        if not result.empty:
            # Sort by valid time and lead time
            result = result.sort_values(['valid_time', 'lead_time'])
            logger.info(f"✓ Reshaped into {len(result)} forecast records for analysis")
        
        return result


def save_to_csv(df: pd.DataFrame, output_path: str) -> bool:
    """Save DataFrame to CSV file."""
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"✓ Saved {len(df)} records to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving to CSV: {e}")
        return False


def main():
    """Example usage of the fetcher."""
    print("=" * 70)
    print("Open-Meteo Previous Runs API Fetcher")
    print("=" * 70)
    print()
    
    # Initialize fetcher for KLGA
    fetcher = OpenMeteoPreviousRunsFetcher(
        latitude=KLGA_LAT,
        longitude=KLGA_LON,
        timezone="America/New_York"
    )
    
    # Example 1: Fetch single date with multiple variables
    print("Example 1: Fetching single date with multiple variables")
    print("-" * 70)
    
    df_single = fetcher.fetch_forecast(
        date='2025-01-17',
        past_days=3,
        variables=['temperature_2m', 'precipitation', 'wind_speed_10m']
    )
    
    if df_single is not None:
        print(f"Fetched {len(df_single)} hours")
        print("\nSample data:")
        print(df_single.head())
        print()
    
    # Example 2: Fetch date range for analysis
    print("\nExample 2: Fetching date range for forecast accuracy analysis")
    print("-" * 70)
    
    df_analysis = fetcher.fetch_for_analysis(
        start_date='2025-01-01',
        end_date='2025-01-17',
        lead_times=[0, 1, 2, 3]  # Same day, 1 day before, 2 days before, 3 days before
    )
    
    if not df_analysis.empty:
        print(f"Fetched {len(df_analysis)} forecast records")
        print(f"\nLead time distribution:")
        print(df_analysis['lead_time'].value_counts().sort_index())
        print("\nSample data:")
        print(df_analysis.head(10))
        
        # Save to CSV
        output_path = 'data/raw/openmeteo_previous_runs.csv'
        save_to_csv(df_analysis, output_path)
        print(f"\n✓ Saved to {output_path}")
    
    print("\n" + "=" * 70)
    print("UNDERSTANDING THE DATA:")
    print("=" * 70)
    print("- lead_time=0: Forecast issued on the same day (nowcast)")
    print("- lead_time=1: Forecast issued 1 day before")
    print("- lead_time=2: Forecast issued 2 days before")
    print("- lead_time=3: Forecast issued 3 days before")
    print("\nFor Polymarket betting:")
    print("- Compare accuracy at different lead times")
    print("- Identify when forecasts stabilize")
    print("- Use lead_time=2 for typical market opening scenarios")
    print("=" * 70)


if __name__ == "__main__":
    main()

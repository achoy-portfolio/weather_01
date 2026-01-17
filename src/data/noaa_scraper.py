"""NOAA Climate Data Online (CDO) scraper for historical weather data.

Get a free API key at: https://www.ncdc.noaa.gov/cdo-web/token
"""

import time
from datetime import date, timedelta
from typing import Optional

import pandas as pd
import requests


class NOAADataError(Exception):
    """Raised when NOAA data cannot be fetched."""
    pass


class NOAAScraper:
    """
    Fetches historical weather data from NOAA Climate Data Online API.
    
    This provides access to decades of historical weather data.
    Requires a free API key from: https://www.ncdc.noaa.gov/cdo-web/token
    
    Rate limit: 5 requests per second, 1000 requests per day.
    """

    API_URL = "https://www.ncei.noaa.gov/cdo-web/api/v2/data"
    
    # NYC area GHCND station IDs
    STATIONS = {
        "LGA": "GHCND:USW00014732",      # LaGuardia Airport
        "JFK": "GHCND:USW00094789",      # JFK Airport
        "CENTRAL_PARK": "GHCND:USW00094728",  # Central Park
    }
    
    def __init__(self, api_key: str, station_id: str = "LGA"):
        """
        Initialize NOAA scraper.
        
        Args:
            api_key: NOAA CDO API key (get free at https://www.ncdc.noaa.gov/cdo-web/token)
            station_id: Station identifier (LGA, JFK, or CENTRAL_PARK)
        """
        if station_id not in self.STATIONS:
            raise ValueError(f"Unknown station: {station_id}. Use one of: {list(self.STATIONS.keys())}")
        
        self.api_key = api_key
        self.station_id = station_id
        self.ghcnd_id = self.STATIONS[station_id]
        self._last_request_time: Optional[float] = None
        self._rate_limit_delay = 0.25  # 5 requests/sec max
        
        self.session = requests.Session()
        self.session.headers.update({
            'token': api_key,
        })

    def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        if self._last_request_time is not None:
            elapsed = time.time() - self._last_request_time
            if elapsed < self._rate_limit_delay:
                time.sleep(self._rate_limit_delay - elapsed)
        self._last_request_time = time.time()

    def _tenths_to_fahrenheit(self, tenths_celsius: Optional[float]) -> Optional[float]:
        """Convert tenths of Celsius to Fahrenheit."""
        if tenths_celsius is None:
            return None
        celsius = tenths_celsius / 10.0
        return round(celsius * 9/5 + 32, 1)

    def _tenths_mm_to_inches(self, tenths_mm: Optional[float]) -> Optional[float]:
        """Convert tenths of mm to inches."""
        if tenths_mm is None:
            return 0.0
        return round((tenths_mm / 10.0) * 0.0393701, 2)

    def fetch_date_range(self, start_date: date, end_date: date,
                         progress_callback=None) -> pd.DataFrame:
        """
        Fetch weather data for a date range.
        
        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)  
            progress_callback: Optional callback function(current, total, message)
            
        Returns:
            DataFrame with daily weather observations
        """
        # NOAA API limits to 1 year per request
        all_results = []
        current_start = start_date
        
        while current_start <= end_date:
            # Chunk into 1-year periods
            current_end = min(current_start + timedelta(days=365), end_date)
            
            print(f"Fetching {current_start} to {current_end}...")
            
            chunk_data = self._fetch_chunk(current_start, current_end)
            all_results.extend(chunk_data)
            
            current_start = current_end + timedelta(days=1)
        
        if not all_results:
            return pd.DataFrame(columns=['date', 'max_temp', 'min_temp', 'avg_humidity', 
                                        'avg_wind_speed', 'total_precipitation'])
        
        # Process results into daily records
        df = self._process_results(all_results, start_date, end_date)
        
        print(f"Completed: {len(df)} days, {df['max_temp'].notna().sum()} with valid data")
        
        return df

    def _fetch_chunk(self, start_date: date, end_date: date, max_retries: int = 3) -> list:
        """Fetch a chunk of data (max 1 year)."""
        params = {
            'datasetid': 'GHCND',
            'stationid': self.ghcnd_id,
            'startdate': start_date.isoformat(),
            'enddate': end_date.isoformat(),
            'datatypeid': 'TMAX,TMIN,PRCP,AWND',  # Max temp, min temp, precip, avg wind
            'units': 'standard',
            'limit': 1000,
        }
        
        all_data = []
        offset = 1
        
        while True:
            params['offset'] = offset
            last_error = None
            
            for attempt in range(max_retries):
                try:
                    self._rate_limit()
                    
                    response = self.session.get(self.API_URL, params=params, timeout=30)
                    response.raise_for_status()
                    
                    data = response.json()
                    results = data.get('results', [])
                    
                    if not results:
                        return all_data
                    
                    all_data.extend(results)
                    
                    # Check if more pages
                    metadata = data.get('metadata', {}).get('resultset', {})
                    total_count = metadata.get('count', 0)
                    
                    if offset + len(results) >= total_count:
                        return all_data
                    
                    offset += len(results)
                    break
                    
                except requests.RequestException as e:
                    last_error = e
                    wait_time = 2 ** (attempt + 1)
                    print(f"  Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
            else:
                raise NOAADataError(f"Failed after {max_retries} attempts: {last_error}")
        
        return all_data

    def _process_results(self, results: list, start_date: date, end_date: date) -> pd.DataFrame:
        """Process raw API results into a daily DataFrame."""
        # Group by date
        daily_data = {}
        
        for r in results:
            obs_date = date.fromisoformat(r['date'][:10])
            if obs_date not in daily_data:
                daily_data[obs_date] = {
                    'date': obs_date,
                    'max_temp': None,
                    'min_temp': None,
                    'avg_humidity': None,  # Not available in GHCND
                    'avg_wind_speed': None,
                    'total_precipitation': None,
                }
            
            datatype = r['datatype']
            value = r['value']
            
            if datatype == 'TMAX':
                daily_data[obs_date]['max_temp'] = self._tenths_to_fahrenheit(value)
            elif datatype == 'TMIN':
                daily_data[obs_date]['min_temp'] = self._tenths_to_fahrenheit(value)
            elif datatype == 'PRCP':
                daily_data[obs_date]['total_precipitation'] = self._tenths_mm_to_inches(value)
            elif datatype == 'AWND':
                # AWND is in tenths of m/s, convert to mph
                if value is not None:
                    mps = value / 10.0
                    daily_data[obs_date]['avg_wind_speed'] = round(mps * 2.237, 1)
        
        # Fill in missing dates
        current = start_date
        while current <= end_date:
            if current not in daily_data:
                daily_data[current] = {
                    'date': current,
                    'max_temp': None,
                    'min_temp': None,
                    'avg_humidity': None,
                    'avg_wind_speed': None,
                    'total_precipitation': None,
                }
            current += timedelta(days=1)
        
        # Convert to DataFrame
        df = pd.DataFrame(list(daily_data.values()))
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        df.set_index('date', inplace=True)
        
        return df

    def save_to_csv(self, df: pd.DataFrame, filepath: str) -> None:
        """Save weather data to CSV file."""
        df.to_csv(filepath, index=True)
        print(f"Saved weather data to {filepath}")

    def load_from_csv(self, filepath: str) -> pd.DataFrame:
        """Load weather data from CSV file."""
        df = pd.read_csv(filepath, parse_dates=['date'], index_col='date')
        return df

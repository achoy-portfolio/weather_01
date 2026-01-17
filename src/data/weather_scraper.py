"""National Weather Service (NWS) scraper for historical weather data."""

import time
from datetime import date, timedelta, datetime
from typing import Optional

import pandas as pd
import requests


class WeatherDataError(Exception):
    """Raised when weather data cannot be fetched."""
    pass


class WeatherScraper:
    """
    Fetches historical weather data from National Weather Service API.
    
    Note: NWS API only retains ~7 days of observations. For older historical
    data, consider using NOAA's Climate Data Online (CDO) or other sources.
    """

    # NWS API endpoints
    STATIONS_URL = "https://api.weather.gov/stations/{station}/observations"
    
    # NYC area stations
    STATIONS = {
        "KLGA": "KLGA",  # LaGuardia Airport
        "KJFK": "KJFK",  # JFK Airport
        "KNYC": "KNYC",  # Central Park
    }
    
    def __init__(self, station_id: str = "KLGA"):
        """
        Initialize scraper with weather station ID.
        
        Args:
            station_id: Weather station identifier (default: KLGA for LaGuardia)
        """
        self.station_id = station_id
        self._last_request_time: Optional[float] = None
        self._rate_limit_delay = 0.5  # NWS has no limit, but be polite
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': '(NYC Weather App, contact@example.com)',
            'Accept': 'application/geo+json',
        })

    def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        if self._last_request_time is not None:
            elapsed = time.time() - self._last_request_time
            if elapsed < self._rate_limit_delay:
                time.sleep(self._rate_limit_delay - elapsed)
        self._last_request_time = time.time()

    def _celsius_to_fahrenheit(self, celsius: Optional[float]) -> Optional[float]:
        """Convert Celsius to Fahrenheit."""
        if celsius is None:
            return None
        return round(celsius * 9/5 + 32, 1)

    def _parse_observations(self, data: dict, target_date: date) -> dict:
        """
        Parse weather observations from NWS API response.
        
        Args:
            data: GeoJSON response from NWS API
            target_date: The date to filter observations for
            
        Returns:
            dict with aggregated daily weather observations
        """
        features = data.get('features', [])
        
        # Filter observations for the target date
        day_obs = []
        for f in features:
            props = f.get('properties', {})
            timestamp = props.get('timestamp')
            if timestamp:
                obs_date = datetime.fromisoformat(timestamp.replace('Z', '+00:00')).date()
                if obs_date == target_date:
                    day_obs.append(props)
        
        if not day_obs:
            return {
                'date': target_date,
                'max_temp': None,
                'min_temp': None,
                'avg_humidity': None,
                'avg_wind_speed': None,
                'total_precipitation': None,
            }
        
        # Extract values (NWS uses metric, convert to imperial)
        temps = []
        humidity = []
        wind_speeds = []
        precip = []
        
        for obs in day_obs:
            # Temperature (Celsius -> Fahrenheit)
            temp_c = obs.get('temperature', {}).get('value')
            if temp_c is not None:
                temps.append(self._celsius_to_fahrenheit(temp_c))
            
            # Relative humidity (%)
            rh = obs.get('relativeHumidity', {}).get('value')
            if rh is not None:
                humidity.append(rh)
            
            # Wind speed (km/h -> mph)
            wind_kmh = obs.get('windSpeed', {}).get('value')
            if wind_kmh is not None:
                wind_speeds.append(round(wind_kmh * 0.621371, 1))
            
            # Precipitation (mm -> inches)
            precip_mm = obs.get('precipitationLastHour', {}).get('value')
            if precip_mm is not None:
                precip.append(precip_mm * 0.0393701)
        
        return {
            'date': target_date,
            'max_temp': max(temps) if temps else None,
            'min_temp': min(temps) if temps else None,
            'avg_humidity': round(sum(humidity) / len(humidity), 1) if humidity else None,
            'avg_wind_speed': round(sum(wind_speeds) / len(wind_speeds), 1) if wind_speeds else None,
            'total_precipitation': round(sum(precip), 2) if precip else 0.0,
        }

    def fetch_daily_history(self, target_date: date, max_retries: int = 3) -> dict:
        """
        Fetch weather data for a specific date.
        
        Args:
            target_date: The date to fetch weather data for
            max_retries: Maximum number of retry attempts
            
        Returns:
            dict with keys: date, max_temp, min_temp, avg_humidity, avg_wind_speed, total_precipitation
            
        Raises:
            WeatherDataError: If data cannot be fetched after all retries
        """
        url = self.STATIONS_URL.format(station=self.station_id)
        
        # NWS API uses ISO 8601 date format
        start_dt = datetime.combine(target_date, datetime.min.time())
        end_dt = datetime.combine(target_date + timedelta(days=1), datetime.min.time())
        
        params = {
            'start': start_dt.isoformat() + 'Z',
            'end': end_dt.isoformat() + 'Z',
        }
        
        last_error = None
        
        for attempt in range(max_retries):
            try:
                self._rate_limit()
                
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                return self._parse_observations(data, target_date)
                
            except requests.RequestException as e:
                last_error = e
                wait_time = 2 ** (attempt + 1)
                print(f"Attempt {attempt + 1} failed for {target_date}: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                
            except (KeyError, ValueError) as e:
                last_error = e
                wait_time = 2 ** (attempt + 1)
                print(f"Parse error on attempt {attempt + 1} for {target_date}: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
        
        raise WeatherDataError(f"Failed to fetch data for {target_date} after {max_retries} attempts: {last_error}")

    def fetch_date_range(self, start_date: date, end_date: date, 
                         progress_callback=None) -> pd.DataFrame:
        """
        Fetch weather data for a date range.
        
        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            progress_callback: Optional callback function(current, total, date)
            
        Returns:
            DataFrame with daily weather observations
        """
        results = []
        current_date = start_date
        total_days = (end_date - start_date).days + 1
        day_count = 0
        
        print(f"Fetching weather data from {start_date} to {end_date} ({total_days} days)...")
        
        while current_date <= end_date:
            day_count += 1
            
            try:
                data = self.fetch_daily_history(current_date)
                results.append(data)
                
                if progress_callback:
                    progress_callback(day_count, total_days, current_date)
                else:
                    print(f"  [{day_count}/{total_days}] {current_date}: "
                          f"High={data['max_temp']}°F, Low={data['min_temp']}°F")
                    
            except WeatherDataError as e:
                print(f"  [{day_count}/{total_days}] Failed {current_date}: {e}")
                results.append({
                    'date': current_date,
                    'max_temp': None,
                    'min_temp': None,
                    'avg_humidity': None,
                    'avg_wind_speed': None,
                    'total_precipitation': None,
                })
            
            current_date += timedelta(days=1)
        
        df = pd.DataFrame(results)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        print(f"Completed: {len(results)} days, {df['max_temp'].notna().sum()} with valid data")
        
        return df

    def fetch_raw_observations(self, start_datetime: datetime, end_datetime: datetime,
                                max_retries: int = 3) -> pd.DataFrame:
        """
        Fetch raw 5-minute observations for a datetime range.
        
        Args:
            start_datetime: Start datetime (inclusive)
            end_datetime: End datetime (inclusive)
            max_retries: Maximum number of retry attempts
            
        Returns:
            DataFrame with 5-minute interval observations including:
            timestamp, temp, humidity, wind_speed, wind_direction, pressure, precipitation
        """
        url = self.STATIONS_URL.format(station=self.station_id)
        
        params = {
            'start': start_datetime.isoformat() + 'Z' if start_datetime.tzinfo is None else start_datetime.isoformat(),
            'end': end_datetime.isoformat() + 'Z' if end_datetime.tzinfo is None else end_datetime.isoformat(),
        }
        
        last_error = None
        
        for attempt in range(max_retries):
            try:
                self._rate_limit()
                
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                return self._parse_raw_observations(data)
                
            except requests.RequestException as e:
                last_error = e
                wait_time = 2 ** (attempt + 1)
                print(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
        
        raise WeatherDataError(f"Failed to fetch observations after {max_retries} attempts: {last_error}")

    def _parse_raw_observations(self, data: dict) -> pd.DataFrame:
        """Parse raw observations into a DataFrame."""
        features = data.get('features', [])
        
        records = []
        for f in features:
            props = f.get('properties', {})
            
            timestamp = props.get('timestamp')
            if not timestamp:
                continue
            
            # Parse timestamp
            ts = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            
            # Extract all available fields
            temp_c = props.get('temperature', {}).get('value')
            humidity = props.get('relativeHumidity', {}).get('value')
            wind_kmh = props.get('windSpeed', {}).get('value')
            wind_dir = props.get('windDirection', {}).get('value')
            pressure_pa = props.get('barometricPressure', {}).get('value')
            precip_mm = props.get('precipitationLastHour', {}).get('value')
            visibility_m = props.get('visibility', {}).get('value')
            dewpoint_c = props.get('dewpoint', {}).get('value')
            
            records.append({
                'timestamp': ts,
                'temp_f': self._celsius_to_fahrenheit(temp_c),
                'humidity': round(humidity, 1) if humidity else None,
                'wind_speed_mph': round(wind_kmh * 0.621371, 1) if wind_kmh else None,
                'wind_direction': wind_dir,
                'pressure_inhg': round(pressure_pa * 0.0002953, 2) if pressure_pa else None,
                'precipitation_in': round(precip_mm * 0.0393701, 2) if precip_mm else None,
                'visibility_mi': round(visibility_m * 0.000621371, 1) if visibility_m else None,
                'dewpoint_f': self._celsius_to_fahrenheit(dewpoint_c),
            })
        
        df = pd.DataFrame(records)
        if not df.empty:
            df = df.sort_values('timestamp')
            df.set_index('timestamp', inplace=True)
        
        return df

    def save_to_csv(self, df: pd.DataFrame, filepath: str) -> None:
        """Save weather data to CSV file."""
        df.to_csv(filepath, index=True)
        print(f"Saved weather data to {filepath}")

    def load_from_csv(self, filepath: str) -> pd.DataFrame:
        """Load weather data from CSV file."""
        # Detect index column name
        df = pd.read_csv(filepath, nrows=1)
        index_col = 'timestamp' if 'timestamp' in df.columns else 'date'
        df = pd.read_csv(filepath, parse_dates=[index_col], index_col=index_col)
        return df

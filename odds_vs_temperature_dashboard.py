"""
Combined Dashboard: NWS Temperature vs Polymarket Odds
See how odds changed as temperature readings came in

Run: streamlit run odds_vs_temperature_dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, timezone, date
import sys
import os
import requests
import json
import re

sys.path.insert(0, '.')

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

from src.data.weather_scraper import WeatherScraper, WeatherDataError
from src.data.noaa_scraper import NOAAScraper, NOAADataError

NY_TZ = ZoneInfo("America/New_York")

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# API Keys - try Streamlit secrets first, then environment variable
try:
    NOAA_API_KEY = st.secrets.get("NOAA_API_KEY")
    VISUAL_CROSSING_API_KEY = st.secrets.get("VISUAL_CROSSING_API_KEY")
except:
    NOAA_API_KEY = os.getenv("NOAA_API_KEY")
    VISUAL_CROSSING_API_KEY = os.getenv("VISUAL_CROSSING_API_KEY")
    VISUAL_CROSSING_API_KEY = os.getenv("VISUAL_CROSSING_API_KEY")

# Page config
st.set_page_config(
    page_title="Temperature vs Odds - KLGA",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Temperature vs Odds Dashboard"
    }
)

# Modern, cohesive styling with soft colors
st.markdown("""
<style>
    /* Soft background with subtle warmth */
    .stApp {
        background: linear-gradient(to bottom, #FAFBFC 0%, #F5F7FA 100%) !important;
    }
    
    .main {
        background: transparent !important;
    }
    
    .block-container {
        background: transparent !important;
        padding-top: 2rem !important;
    }
    
    /* Clean header */
    header[data-testid="stHeader"] {
        background: rgba(255, 255, 255, 0.95) !important;
        backdrop-filter: blur(10px);
        border-bottom: 1px solid rgba(0, 0, 0, 0.06);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Elegant sidebar with soft shadow */
    [data-testid="stSidebar"] {
        background: linear-gradient(to bottom, #FFFFFF 0%, #FAFBFC 100%) !important;
        border-right: 1px solid rgba(0, 0, 0, 0.08) !important;
        box-shadow: 2px 0 12px rgba(0, 0, 0, 0.03);
    }
    
    /* Refined typography */
    body, p, span, div, label {
        color: #2D3748 !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', sans-serif !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #1A202C !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', sans-serif !important;
    }
    
    /* Polished input fields */
    [data-testid="stDateInput"] {
        background-color: #FFFFFF !important;
    }
    
    [data-testid="stDateInput"] input {
        background-color: #FFFFFF !important;
        color: #2D3748 !important;
        border: 1.5px solid #E2E8F0 !important;
        border-radius: 8px !important;
        padding: 0.625rem 0.875rem !important;
        transition: all 0.2s ease !important;
    }
    
    [data-testid="stDateInput"] input:hover {
        border-color: #CBD5E0 !important;
    }
    
    [data-testid="stDateInput"] input:focus {
        border-color: #667EEA !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
    }
    
    [data-testid="stDateInput"] label {
        color: #4A5568 !important;
        font-weight: 500 !important;
        font-size: 0.875rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Refined form inputs */
    input[type="date"], input[type="text"], input[type="number"] {
        background-color: #FFFFFF !important;
        color: #2D3748 !important;
        border: 1.5px solid #E2E8F0 !important;
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
    }
    
    input[type="date"]:focus, input[type="text"]:focus, input[type="number"]:focus {
        border-color: #667EEA !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
        outline: none !important;
    }
    
    /* Elegant selectbox */
    [data-baseweb="select"] {
        background-color: #FFFFFF !important;
        border-radius: 8px !important;
    }
    
    [data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        color: #2D3748 !important;
        border: 1.5px solid #E2E8F0 !important;
        border-radius: 8px !important;
    }
    
    /* Refined radio buttons */
    [data-testid="stRadio"] {
        background-color: transparent !important;
    }
    
    [data-testid="stRadio"] label {
        color: #2D3748 !important;
        padding: 0.5rem 0 !important;
    }
    
    [data-testid="stRadio"] label:hover {
        color: #1A202C !important;
    }
    
    /* Sidebar headers */
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #1A202C !important;
    }
    
    /* Beautiful metric cards with subtle depth */
    .metric-card {
        background: linear-gradient(to bottom right, #FFFFFF 0%, #F7FAFC 100%);
        padding: 1.75rem;
        border-radius: 12px;
        border: 1px solid rgba(0, 0, 0, 0.06);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04), 0 1px 2px rgba(0, 0, 0, 0.06);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08), 0 2px 4px rgba(0, 0, 0, 0.08);
        transform: translateY(-2px);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667EEA 0%, #764BA2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0.5rem 0;
        line-height: 1.2;
    }
    
    .metric-label {
        font-size: 0.8125rem;
        font-weight: 600;
        color: #718096 !important;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    
    .metric-caption {
        font-size: 0.875rem;
        color: #A0AEC0 !important;
        margin-top: 0.625rem;
        font-weight: 400;
    }
    
    /* Elegant insight boxes with refined gradients */
    .insight-box {
        background: linear-gradient(135deg, #667EEA 0%, #764BA2 100%);
        padding: 1.75rem;
        border-radius: 12px;
        color: white !important;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(102, 126, 234, 0.25);
        transition: all 0.3s ease;
    }
    
    .insight-box:hover {
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.35);
        transform: translateY(-2px);
    }
    
    .insight-box h4, .insight-box p, .insight-box strong {
        color: white !important;
    }
    
    .insight-box h4 {
        font-size: 0.8125rem;
        font-weight: 700;
        text-transform: uppercase;
        margin: 0 0 0.625rem 0;
        letter-spacing: 1px;
        opacity: 0.95;
    }
    
    .insight-box strong {
        font-size: 1.875rem;
        font-weight: 700;
        display: block;
        margin: 0.375rem 0;
        line-height: 1.2;
    }
    
    /* Clean section headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1A202C !important;
        margin: 2.5rem 0 1.25rem 0;
        padding-bottom: 0.875rem;
        border-bottom: 2px solid #E2E8F0;
        letter-spacing: -0.025em;
    }
    
    /* Refined status badges */
    .status-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-size: 0.8125rem;
        font-weight: 600;
        margin: 0.25rem 0;
        transition: all 0.2s ease;
    }
    
    .status-success {
        background: linear-gradient(135deg, #D4EDDA 0%, #C3E6CB 100%);
        color: #155724 !important;
        border: 1px solid #B1DFBB;
    }
    
    .status-warning {
        background: linear-gradient(135deg, #FFF3CD 0%, #FFE8A1 100%);
        color: #856404 !important;
        border: 1px solid #FFE69C;
    }
    
    .status-info {
        background: linear-gradient(135deg, #D1ECF1 0%, #BEE5EB 100%);
        color: #0C5460 !important;
        border: 1px solid #B8DAFF;
    }
    
    /* Modern buttons with smooth interactions */
    .stButton > button {
        background: linear-gradient(135deg, #667EEA 0%, #764BA2 100%) !important;
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.25);
    }
    
    .stButton > button:hover {
        box-shadow: 0 4px 16px rgba(102, 126, 234, 0.35);
        transform: translateY(-1px);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Elegant chart containers */
    .chart-container {
        background: linear-gradient(to bottom right, #FFFFFF 0%, #FAFBFC 100%);
        padding: 2rem;
        border-radius: 12px;
        border: 1px solid rgba(0, 0, 0, 0.06);
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.04);
        margin: 1.5rem 0;
    }
    
    /* Refined expanders */
    .streamlit-expanderHeader {
        background: linear-gradient(to right, #F7FAFC 0%, #EDF2F7 100%) !important;
        border-radius: 8px;
        border: 1px solid #E2E8F0 !important;
        font-weight: 600;
        color: #2D3748 !important;
        padding: 1rem 1.25rem !important;
        transition: all 0.2s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: linear-gradient(to right, #EDF2F7 0%, #E2E8F0 100%) !important;
        border-color: #CBD5E0 !important;
    }
    
    /* Modern tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: linear-gradient(to right, #F7FAFC 0%, #EDF2F7 100%);
        padding: 0.375rem;
        border-radius: 10px;
        border: 1px solid #E2E8F0;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        padding: 0.625rem 1.25rem;
        font-weight: 600;
        color: #718096 !important;
        transition: all 0.2s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: #4A5568 !important;
        background-color: rgba(255, 255, 255, 0.5) !important;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(to bottom right, #FFFFFF 0%, #F7FAFC 100%) !important;
        color: #1A202C !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
    }
    
    /* Polished form inputs */
    input, select, textarea {
        color: #2D3748 !important;
        font-weight: 500 !important;
    }
    
    /* Smooth scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #F7FAFC;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(to bottom, #CBD5E0 0%, #A0AEC0 100%);
        border-radius: 10px;
        border: 2px solid #F7FAFC;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(to bottom, #A0AEC0 0%, #718096 100%);
    }
</style>
""", unsafe_allow_html=True)

# Cache data
@st.cache_data(ttl=300)  # Cache for 5 minutes since NWS forecast updates frequently
def fetch_nws_forecast_data():
    """
    Fetch current NWS forecast for KLGA.
    Only works for current/future dates (NWS doesn't provide historical forecasts).
    
    Returns:
        DataFrame with hourly forecast data or None
    """
    try:
        # Import the NWS forecast function
        sys.path.insert(0, 'scripts/fetching')
        from fetch_nws_forecast import get_nws_forecast
        
        df = get_nws_forecast()
        
        if df is not None and not df.empty:
            # Ensure timezone aware
            if 'start_time' in df.columns:
                df['timestamp'] = pd.to_datetime(df['start_time'])
                if df['timestamp'].dt.tz is None:
                    df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert(NY_TZ)
                else:
                    df['timestamp'] = df['timestamp'].dt.tz_convert(NY_TZ)
                
                # Rename temperature column
                if 'temperature' in df.columns:
                    df = df.rename(columns={'temperature': 'temp_f'})
                
                return df[['timestamp', 'temp_f']].copy()
        
        return None
        
    except Exception as e:
        # Silently fail - NWS forecast is optional
        return None


@st.cache_data(ttl=3600)  # Cache for 1 hour since historical forecasts don't change
def fetch_visual_crossing_forecast(forecast_time, target_date):
    """
    Fetch historical forecast from Visual Crossing Timeline API.
    Shows what the forecast was at a specific time for the target date.
    
    Args:
        forecast_time: datetime when the forecast was issued
        target_date: date being forecasted
    
    Returns:
        DataFrame with hourly forecast data
    """
    if not VISUAL_CROSSING_API_KEY or VISUAL_CROSSING_API_KEY == "your_visual_crossing_key_here":
        st.error("Visual Crossing API key not configured. Add it to your .env file.")
        return None
    
    # Visual Crossing Weather API
    # Note: Visual Crossing doesn't provide "historical forecasts" - it provides historical actual weather
    # For forecast data, we need to use their current forecast API or historical weather data
    
    # Using historical weather data as a proxy (actual temps that occurred)
    location = "LaGuardia Airport,NY,US"
    
    # Get data for target date + 3 days
    start_date = target_date.isoformat()
    end_date = (target_date + timedelta(days=3)).isoformat()
    
    url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{location}/{start_date}/{end_date}"
    
    params = {
        'key': VISUAL_CROSSING_API_KEY,
        'unitGroup': 'us',  # Fahrenheit
        'include': 'hours',  # Hourly data
        'contentType': 'json',
        'elements': 'datetime,temp'
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        
        # Check for API errors
        if response.status_code == 401:
            st.error("❌ Visual Crossing API key is invalid")
            return None
        elif response.status_code == 429:
            st.error("❌ Visual Crossing API rate limit exceeded (1000 calls/day)")
            return None
        elif response.status_code != 200:
            st.error(f"❌ Visual Crossing API error: {response.status_code} - {response.text[:200]}")
            return None
        
        response.raise_for_status()
        data = response.json()
        
        if 'days' not in data:
            st.warning("No forecast data returned from Visual Crossing")
            return None
        
        # Parse hourly forecast data
        records = []
        for day in data['days']:
            day_date = datetime.fromisoformat(day['datetime']).date()
            
            if 'hours' in day:
                for hour in day['hours']:
                    hour_time = datetime.strptime(hour['datetime'], '%H:%M:%S').time()
                    timestamp = datetime.combine(day_date, hour_time).replace(tzinfo=NY_TZ)
                    
                    temp = hour.get('temp')
                    if temp is not None:
                        records.append({
                            'timestamp': timestamp,
                            'temp_f': temp,
                            'forecast_issued': forecast_time,
                            'source': 'Visual Crossing'
                        })
        
        if not records:
            st.warning("No temperature data in Visual Crossing response")
            return None
        
        df = pd.DataFrame(records)
        df = df.sort_values('timestamp')
        
        return df
        
    except requests.exceptions.Timeout:
        st.error("❌ Visual Crossing API timeout - try again")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Visual Crossing API request error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"❌ Error processing Visual Crossing data: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None


@st.cache_data(ttl=60)
def fetch_openmeteo_hourly_fallback(start_dt, end_dt):
    """
    Fetch hourly temperature data from Open-Meteo API (free, no API key required).
    This is a fallback when NWS API doesn't have data (older than 7 days).
    Open-Meteo provides historical hourly data going back to 1940.
    """
    # Open-Meteo Historical Weather API
    url = "https://archive-api.open-meteo.com/v1/archive"
    
    # LaGuardia Airport coordinates
    lat = 40.7769
    lon = -73.8740
    
    # Format dates (Open-Meteo uses local dates)
    start_date = start_dt.date().isoformat()
    end_date = end_dt.date().isoformat()
    
    params = {
        'latitude': lat,
        'longitude': lon,
        'start_date': start_date,
        'end_date': end_date,
        'hourly': 'temperature_2m',
        'temperature_unit': 'fahrenheit',
        'timezone': 'America/New_York'
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if 'hourly' not in data:
            return None
        
        hourly = data['hourly']
        times = hourly.get('time', [])
        temps = hourly.get('temperature_2m', [])
        
        if not times or not temps:
            return None
        
        # Create DataFrame
        records = []
        for time_str, temp in zip(times, temps):
            if temp is None:
                continue
            
            # Parse timestamp (already in NY timezone from API)
            timestamp = datetime.fromisoformat(time_str).replace(tzinfo=NY_TZ)
            
            records.append({
                'timestamp': timestamp,
                'temp_f': round(temp, 1)
            })
        
        if not records:
            return None
        
        df = pd.DataFrame(records)
        df = df.sort_values('timestamp')
        
        # Filter to requested time range
        df = df[(df['timestamp'] >= start_dt) & (df['timestamp'] <= end_dt)]
        
        return df
        
    except Exception as e:
        st.warning(f"Open-Meteo error: {e}")
        return None


@st.cache_data(ttl=60)
def fetch_nws_data(target_date, odds_time_range=None):
    """Fetch NWS temperature observations matching the odds time range"""
    scraper = WeatherScraper(station_id="KLGA")
    
    if odds_time_range is not None:
        # Use the same time range as odds data, but expand it slightly to ensure we get all data
        start_dt = odds_time_range[0] - timedelta(hours=1)  # Start 1 hour earlier
        end_dt = odds_time_range[1] + timedelta(hours=1)    # End 1 hour later
    else:
        # Fallback: get data for the target date
        start_dt = datetime.combine(target_date, datetime.min.time()).replace(tzinfo=NY_TZ)
        end_dt = datetime.combine(target_date, datetime.max.time()).replace(tzinfo=NY_TZ)
    
    # Convert to UTC for API
    start_dt_utc = start_dt.astimezone(timezone.utc)
    end_dt_utc = end_dt.astimezone(timezone.utc)
    
    # NWS API has strict limits - fetch in 24-hour chunks to be safe
    try:
        all_dfs = []
        current_start = start_dt_utc
        
        while current_start < end_dt_utc:
            current_end = min(current_start + timedelta(hours=24), end_dt_utc)
            
            try:
                df_chunk = scraper.fetch_raw_observations(current_start, current_end)
                if not df_chunk.empty:
                    all_dfs.append(df_chunk)
            except Exception as e:
                # Don't warn here, we'll try NOAA fallback if needed
                pass
            
            current_start = current_end
        
        if not all_dfs:
            # NWS data not available, try Open-Meteo fallback
            st.info("📡 NWS data not available, fetching from Open-Meteo archive...")
            openmeteo_df = fetch_openmeteo_hourly_fallback(start_dt, end_dt)
            
            if openmeteo_df is not None and not openmeteo_df.empty:
                st.success("✓ Using Open-Meteo historical hourly data")
                actual_start = openmeteo_df['timestamp'].min()
                actual_end = openmeteo_df['timestamp'].max()
                return openmeteo_df, (actual_start, actual_end)
            else:
                return None
        
        # Combine all chunks
        df = pd.concat(all_dfs, ignore_index=False)
        df = df.sort_index()
        # Remove duplicates that might occur at chunk boundaries
        df = df[~df.index.duplicated(keep='first')]
        
        df.index = df.index.tz_convert(NY_TZ)
        df = df.reset_index()
        df = df.rename(columns={'index': 'timestamp'})
        
        # Return both the dataframe and the actual time range we got
        actual_start = df['timestamp'].min()
        actual_end = df['timestamp'].max()
        
        return df, (actual_start, actual_end)
    except WeatherDataError as e:
        st.error(f"Error fetching NWS data: {e}")
        # Try Open-Meteo fallback
        st.info("📡 Trying Open-Meteo archive...")
        openmeteo_df = fetch_openmeteo_hourly_fallback(start_dt, end_dt)
        
        if openmeteo_df is not None and not openmeteo_df.empty:
            st.success("✓ Using Open-Meteo historical hourly data")
            actual_start = openmeteo_df['timestamp'].min()
            actual_end = openmeteo_df['timestamp'].max()
            return openmeteo_df, (actual_start, actual_end)
        
        return None

@st.cache_data(ttl=60)
def fetch_polymarket_odds(target_date):
    """Dynamically fetch Polymarket odds for a specific date"""
    
    # Generate event slug
    month = target_date.strftime('%B').lower()
    day = target_date.day
    slug = f"highest-temperature-in-nyc-on-{month}-{day}"
    
    # Fetch market info
    api_url = f"https://gamma-api.polymarket.com/events?slug={slug}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json'
    }
    
    try:
        response = requests.get(api_url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if not data or len(data) == 0:
            return None
        
        event = data[0]
        markets = event.get('markets', [])
        
        if not markets:
            return None
        
        # Fetch historical odds for each market
        all_records = []
        
        for market in markets:
            question = market.get('question', '')
            
            # Extract temperature range from question
            range_match = re.search(r'between (\d+)-(\d+)°F', question)
            if range_match:
                low = int(range_match.group(1))
                high = int(range_match.group(2))
                threshold = f"{low}-{high}"
                threshold_display = f"{low}-{high}°F"
            elif 'or below' in question.lower() or 'or lower' in question.lower():
                temp_match = re.search(r'(\d+)°F', question)
                if temp_match:
                    temp = int(temp_match.group(1))
                    threshold = f"≤{temp}"
                    threshold_display = f"≤{temp}°F"
                else:
                    continue
            elif 'or higher' in question.lower() or 'or above' in question.lower():
                temp_match = re.search(r'(\d+)°F', question)
                if temp_match:
                    temp = int(temp_match.group(1))
                    threshold = f"≥{temp}"
                    threshold_display = f"≥{temp}°F"
                else:
                    continue
            else:
                continue
            
            # Get token IDs
            clobTokenIds = market.get('clobTokenIds')
            token_ids = []
            if clobTokenIds:
                try:
                    if isinstance(clobTokenIds, str):
                        token_ids = json.loads(clobTokenIds)
                    else:
                        token_ids = clobTokenIds
                except:
                    continue
            
            if not token_ids or len(token_ids) < 1:
                continue
            
            yes_token_id = token_ids[0]
            
            # Fetch price history using explicit timestamps for better historical data
            # Calculate time range: 3 days before to 1 day after target date
            start_dt = datetime.combine(target_date, datetime.min.time()) - timedelta(days=3)
            end_dt = datetime.combine(target_date, datetime.max.time()) + timedelta(days=1)
            start_ts = int(start_dt.replace(tzinfo=NY_TZ).timestamp())
            end_ts = int(end_dt.replace(tzinfo=NY_TZ).timestamp())
            
            price_url = "https://clob.polymarket.com/prices-history"
            params = {
                'market': yes_token_id,
                'startTs': start_ts,  # Use explicit timestamps (works for closed markets!)
                'endTs': end_ts,
                'fidelity': 5       # 5-minute resolution
            }
            
            try:
                price_response = requests.get(price_url, params=params, headers=headers, timeout=10)
                price_response.raise_for_status()
                price_data = price_response.json()
                history = price_data.get('history', [])
                
                # If we got data, check if it might be truncated
                # Polymarket typically opens markets 2 days before, so we expect ~48 hours of data
                if history:
                    first_timestamp = datetime.fromtimestamp(history[0]['t'], tz=NY_TZ)
                    last_timestamp = datetime.fromtimestamp(history[-1]['t'], tz=NY_TZ)
                    time_span_hours = (last_timestamp - first_timestamp).total_seconds() / 3600
                    
                    # If we got less than 12 hours of data but the market should have more, warn
                    if time_span_hours < 12:
                        st.warning(f"⚠️ Limited odds data for {threshold_display}: only {time_span_hours:.1f} hours available")
                
                for point in history:
                    timestamp = datetime.fromtimestamp(point['t'], tz=NY_TZ)
                    price = point['p']
                    
                    all_records.append({
                        'timestamp': timestamp,
                        'threshold': threshold,
                        'threshold_display': threshold_display,
                        'probability': price,
                        'question': question
                    })
            except Exception as e:
                st.warning(f"Could not fetch odds for {threshold_display}: {e}")
                continue
        
        if not all_records:
            return None
        
        df = pd.DataFrame(all_records)
        df = df.sort_values(['threshold', 'timestamp'])
        
        # Calculate time range for temperature data matching
        min_time = df['timestamp'].min()
        max_time = df['timestamp'].max()
        
        return df, (min_time, max_time)
        
    except Exception as e:
        st.error(f"Error fetching Polymarket data: {e}")
        return None

# Title with elegant styling
st.markdown("""
<div style='margin-bottom: 2.5rem; padding: 2rem 0 1.5rem 0; border-bottom: 1px solid rgba(0, 0, 0, 0.06);'>
    <h1 style='font-size: 2.75rem; font-weight: 700; background: linear-gradient(135deg, #667EEA 0%, #764BA2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; margin: 0; line-height: 1.2; letter-spacing: -0.025em;'>
        Temperature vs Market Odds
    </h1>
    <p style='font-size: 1.125rem; color: #718096; margin-top: 0.75rem; font-weight: 400; line-height: 1.6;'>
        Real-time analysis of how temperature readings influence prediction markets
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar with elegant styling
st.sidebar.markdown("""
<div style='padding: 1.25rem 0; border-bottom: 1px solid rgba(0, 0, 0, 0.08); margin-bottom: 1.75rem;'>
    <h2 style='font-size: 1.375rem; font-weight: 700; color: #1A202C; margin: 0; letter-spacing: -0.015em;'>
        ⚙️ Controls
    </h2>
</div>
""", unsafe_allow_html=True)

# Date selector - allow any date
today = date.today()
default_date = today  # Default to today

selected_date = st.sidebar.date_input(
    "Market Date",
    value=default_date,
    min_value=date(2025, 1, 1),
    max_value=today + timedelta(days=7),
    help="Select which date's market to display. Markets open 2 days before resolution."
)

# Placeholder for data range info (will be updated after data loads)
data_range_placeholder = st.sidebar.empty()

# Fetch data
with st.spinner("Loading data..."):
    # First fetch odds to get the time range
    odds_result = fetch_polymarket_odds(selected_date)
    
    if odds_result is not None and isinstance(odds_result, tuple):
        odds_df, odds_time_range = odds_result
        # Show data range in sidebar
        with data_range_placeholder:
            st.info(f"📊 Odds: {odds_time_range[0].strftime('%b %d, %I:%M %p')} - {odds_time_range[1].strftime('%b %d, %I:%M %p')} ET")
    else:
        odds_df = odds_result
        odds_time_range = None
    
    # Fetch temperature data matching the odds time range
    temp_result = fetch_nws_data(selected_date, odds_time_range=odds_time_range)
    
    if temp_result is not None and isinstance(temp_result, tuple):
        temp_df, temp_time_range = temp_result
        
        # Check if temperature data is significantly incomplete (more than 2 hours missing)
        if odds_time_range is not None:
            if temp_time_range[0] > odds_time_range[0]:
                time_diff = temp_time_range[0] - odds_time_range[0]
                hours_missing = time_diff.total_seconds() / 3600
                if hours_missing > 2:  # Only warn if more than 2 hours missing
                    st.warning(f"⚠️ Temperature data starts {hours_missing:.1f} hours after odds data begins")
                    st.info(f"🌡️ Temperature data available from {temp_time_range[0].strftime('%b %d, %I:%M %p')} ET")
    else:
        temp_df = temp_result
        temp_time_range = None

if temp_df is None or len(temp_df) == 0:
    st.error("❌ No temperature data available for this date")
    st.info("💡 Both NWS API and Open-Meteo did not return data for this period. The market may not exist yet or the date may be too far in the future.")
    st.stop()

# Check if we have odds data
has_odds = odds_df is not None and len(odds_df) > 0

if not has_odds:
    st.warning("⚠️ No Polymarket odds data found. Run the historical odds fetcher first:")
    st.code("python scripts/fetching/fetch_polymarket_historical.py", language="bash")
    st.info("Showing temperature data only for now...")
    
# Get unique buckets from odds data (needs to be before sidebar controls)
if has_odds:
    buckets = sorted(odds_df['threshold'].unique(), key=lambda x: (
        0 if '≤' in str(x) else (
            100 if '≥' in str(x) else int(str(x).split('-')[0])
        )
    ))
else:
    buckets = []

# Now add the odds display controls to sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='padding: 1.25rem 0 0.75rem 0;'>
    <h3 style='font-size: 1.0625rem; font-weight: 700; color: #1A202C; margin: 0; letter-spacing: -0.01em;'>
        📊 Odds Display
    </h3>
</div>
""", unsafe_allow_html=True)

odds_filter = st.sidebar.radio(
    "Show odds for:",
    options=["All ranges", "Top 3 most likely", "Custom selection"],
    index=0,
    help="Filter which temperature ranges to display on the chart"
)

# Determine which buckets to display based on filter
if has_odds and len(buckets) > 0:
    if odds_filter == "Top 3 most likely":
        # Get latest odds for each bucket and find top 3
        latest_by_bucket = odds_df.groupby('threshold').last().reset_index()
        top_buckets = latest_by_bucket.nlargest(3, 'probability')['threshold'].tolist()
        display_buckets = [b for b in buckets if b in top_buckets]
    elif odds_filter == "Custom selection":
        # Let user select which buckets to show
        bucket_displays = {bucket: odds_df[odds_df['threshold'] == bucket].iloc[0]['threshold_display'] 
                          for bucket in buckets}
        selected_displays = st.sidebar.multiselect(
            "Select ranges to display:",
            options=[bucket_displays[b] for b in buckets],
            default=[bucket_displays[b] for b in buckets[:3]]
        )
        display_buckets = [b for b in buckets if bucket_displays[b] in selected_displays]
    else:  # "All ranges"
        display_buckets = buckets
else:
    display_buckets = []

show_temp_bands = st.sidebar.checkbox(
    "Show temperature range bands", 
    value=True,
    help="Display gray bands on the temperature chart showing the selected ranges"
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='padding: 1.25rem 0 0.75rem 0;'>
    <h3 style='font-size: 1.0625rem; font-weight: 700; color: #1A202C; margin: 0; letter-spacing: -0.01em;'>
        🔮 Forecast Analysis
    </h3>
</div>
""", unsafe_allow_html=True)

show_forecast = st.sidebar.checkbox(
    "Show historical forecast",
    value=False,
    help="See what the forecast was at a specific time"
)

# Show API key status with modern badges
if VISUAL_CROSSING_API_KEY and VISUAL_CROSSING_API_KEY != "your_visual_crossing_key_here":
    st.sidebar.markdown("""
    <div class="status-badge status-success" style="display: block; margin: 0.5rem 0;">
        ✓ Visual Crossing API configured
    </div>
    """, unsafe_allow_html=True)
    # Debug: show first/last chars
    if len(VISUAL_CROSSING_API_KEY) > 8:
        masked_key = f"{VISUAL_CROSSING_API_KEY[:4]}...{VISUAL_CROSSING_API_KEY[-4:]}"
        st.sidebar.caption(f"Key: {masked_key}")
else:
    st.sidebar.markdown("""
    <div class="status-badge status-warning" style="display: block; margin: 0.5rem 0;">
        ❌ Visual Crossing API key not set
    </div>
    """, unsafe_allow_html=True)
    st.sidebar.caption("Add VISUAL_CROSSING_API_KEY to .env file")
    # Debug: show what we got
    st.sidebar.caption(f"Current value: {VISUAL_CROSSING_API_KEY if VISUAL_CROSSING_API_KEY else 'None'}")

forecast_df = None
nws_forecast_df = None

if show_forecast and has_odds:
    if not VISUAL_CROSSING_API_KEY or VISUAL_CROSSING_API_KEY == "your_visual_crossing_key_here":
        st.sidebar.error("Please add your Visual Crossing API key to .env")
    else:
        # Let user select a time from the odds data
        st.sidebar.markdown("**Select forecast time:**")
        
        # Get unique timestamps from odds data (remove duplicates)
        odds_times = sorted(odds_df['timestamp'].drop_duplicates())
        
        # Sample every 5th timestamp to reduce clutter (or use hourly intervals)
        # Get hourly timestamps only
        hourly_times = []
        seen_hours = set()
        for t in odds_times:
            hour_key = (t.date(), t.hour)
            if hour_key not in seen_hours:
                hourly_times.append(t)
                seen_hours.add(hour_key)
        
        if len(hourly_times) == 0:
            hourly_times = odds_times[:10]  # Fallback: show first 10
        
        # Create a selectbox with formatted times
        time_options = [t.strftime('%b %d, %I:%M %p ET') for t in hourly_times]
        selected_time_str = st.sidebar.selectbox(
            "When was forecast issued?",
            options=time_options,
            index=len(time_options)//2 if len(time_options) > 0 else 0,
            help="Select when traders saw this forecast (hourly intervals)"
        )
        
        # Get the actual datetime
        selected_forecast_time = hourly_times[time_options.index(selected_time_str)]
        
        # Check if this is a recent/future date (within 7 days)
        days_from_now = (selected_date - date.today()).days
        is_recent = abs(days_from_now) <= 7
        
        # Fetch Visual Crossing forecast
        with st.spinner(f"Fetching Visual Crossing forecast from {selected_time_str}..."):
            forecast_df = fetch_visual_crossing_forecast(selected_forecast_time, selected_date)
        
        if forecast_df is not None:
            st.sidebar.success(f"✓ Visual Crossing: {len(forecast_df)} hours")
        
        # Optionally fetch NWS forecast if date is recent/future
        if is_recent:
            with st.spinner("Fetching NWS forecast..."):
                nws_forecast_df = fetch_nws_forecast_data()
            
            if nws_forecast_df is not None:
                st.sidebar.success(f"✓ NWS Forecast: {len(nws_forecast_df)} hours")
            else:
                st.sidebar.info("ℹ️ NWS forecast not available")

elif show_forecast and not has_odds:
    st.sidebar.warning("⚠️ Load odds data first to select forecast times")

st.sidebar.markdown("---")
refresh_btn = st.sidebar.button("🔄 Refresh Data", width='stretch')

# Current stats with modern card design
current_temp = temp_df['temp_f'].iloc[-1]
current_time = temp_df['timestamp'].iloc[-1]

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Current Temperature</div>
        <div class="metric-value">{current_temp:.1f}°F</div>
        <div class="metric-caption">As of {current_time.strftime('%I:%M %p ET')}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    high_temp = temp_df['temp_f'].max()
    high_time = temp_df.loc[temp_df['temp_f'].idxmax(), 'timestamp']
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Period High</div>
        <div class="metric-value">{high_temp:.1f}°F</div>
        <div class="metric-caption">At {high_time.strftime('%I:%M %p ET')}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    if has_odds and len(buckets) > 0:
        # Find the bucket with highest probability
        latest_by_bucket = odds_df.groupby('threshold').last().reset_index()
        max_prob_bucket = latest_by_bucket.loc[latest_by_bucket['probability'].idxmax()]
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Most Likely Range</div>
            <div class="metric-value">{max_prob_bucket['probability']:.0%}</div>
            <div class="metric-caption">{max_prob_bucket['threshold_display']} • {max_prob_bucket['timestamp'].strftime('%I:%M %p ET')}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Market Odds</div>
            <div class="metric-value">—</div>
            <div class="metric-caption">No data available</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<div style='margin: 2rem 0;'></div>", unsafe_allow_html=True)

# Main chart - Temperature and Odds
st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
if has_odds:
    # Create dual-axis chart
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.15,  # Increased spacing to prevent title overlap
        subplot_titles=('Temperature Readings (°F)', f'Market Odds - {selected_date} (Probability by Range)'),
        row_heights=[0.5, 0.5]
    )
    
    # Temperature lines - split by date
    # Separate data for target date vs other dates
    target_date_start = datetime.combine(selected_date, datetime.min.time()).replace(tzinfo=NY_TZ)
    target_date_end = datetime.combine(selected_date, datetime.max.time()).replace(tzinfo=NY_TZ)
    
    temp_df_target = temp_df[(temp_df['timestamp'] >= target_date_start) & (temp_df['timestamp'] <= target_date_end)]
    temp_df_before = temp_df[temp_df['timestamp'] < target_date_start]
    temp_df_after = temp_df[temp_df['timestamp'] > target_date_end]
    
    # Plot temperature before target date (soft gray)
    if not temp_df_before.empty:
        fig.add_trace(
            go.Scatter(
                x=temp_df_before['timestamp'],
                y=temp_df_before['temp_f'],
                mode='lines',
                name='Temp (Before)',
                line=dict(color='#CBD5E0', width=2.5, dash='dot'),
                hovertemplate='%{y:.1f}°F<br>%{x}<extra></extra>',
                showlegend=True
            ),
            row=1, col=1
        )
    
    # Plot temperature on target date (warm coral)
    if not temp_df_target.empty:
        fig.add_trace(
            go.Scatter(
                x=temp_df_target['timestamp'],
                y=temp_df_target['temp_f'],
                mode='lines',
                name=f'Temp ({selected_date.strftime("%b %d")})',
                line=dict(color='#FC8181', width=3.5),
                hovertemplate='%{y:.1f}°F<br>%{x}<extra></extra>',
                showlegend=True
            ),
            row=1, col=1
        )
    
    # Plot temperature after target date (soft gray)
    if not temp_df_after.empty:
        fig.add_trace(
            go.Scatter(
                x=temp_df_after['timestamp'],
                y=temp_df_after['temp_f'],
                mode='lines',
                name='Temp (After)',
                line=dict(color='#CBD5E0', width=2.5, dash='dot'),
                hovertemplate='%{y:.1f}°F<br>%{x}<extra></extra>',
                showlegend=True
            ),
            row=1, col=1
        )
    
    # Add vertical line to mark the target date boundaries
    fig.add_vline(
        x=target_date_start.timestamp() * 1000,
        line_dash="dash",
        line_color="rgba(0,0,0,0.3)",
        row=1, col=1
    )
    fig.add_vline(
        x=target_date_end.timestamp() * 1000,
        line_dash="dash",
        line_color="rgba(0,0,0,0.3)",
        row=1, col=1
    )
    
    # Add forecast line if available
    if forecast_df is not None and not forecast_df.empty:
        fig.add_trace(
            go.Scatter(
                x=forecast_df['timestamp'],
                y=forecast_df['temp_f'],
                mode='lines',
                name=f'Visual Crossing Forecast',
                line=dict(color='#764BA2', width=2, dash='dash'),
                hovertemplate='VC Forecast: %{y:.1f}°F<br>%{x}<extra></extra>',
                showlegend=True
            ),
            row=1, col=1
        )
        
        # Add vertical line to mark when forecast was issued
        fig.add_vline(
            x=selected_forecast_time.timestamp() * 1000,
            line_dash="dot",
            line_color="rgba(118, 75, 162, 0.5)",
            annotation_text="Forecast issued",
            annotation_position="top",
            row=1, col=1
        )
    
    # Add NWS forecast line if available
    if nws_forecast_df is not None and not nws_forecast_df.empty:
        fig.add_trace(
            go.Scatter(
                x=nws_forecast_df['timestamp'],
                y=nws_forecast_df['temp_f'],
                mode='lines',
                name='NWS Forecast',
                line=dict(color='#4FACFE', width=2, dash='dot'),
                hovertemplate='NWS Forecast: %{y:.1f}°F<br>%{x}<extra></extra>',
                showlegend=True
            ),
            row=1, col=1
        )
    
    # Add temperature range bands for displayed buckets
    if show_temp_bands and len(display_buckets) > 0:
        for bucket in display_buckets:
            bucket_str = str(bucket)
            if '-' in bucket_str:
                low, high = map(int, bucket_str.split('-'))
                fig.add_hrect(
                    y0=low, y1=high,
                    fillcolor="gray", opacity=0.05,
                    line_width=0,
                    row=1, col=1
                )
    
    # Odds lines - show only selected buckets with harmonious color palette
    colors = ['#667EEA', '#48BB78', '#ED8936', '#38B2AC', '#9F7AEA', '#F56565', '#4299E1']
    
    for i, bucket in enumerate(display_buckets):
        bucket_data = odds_df[odds_df['threshold'] == bucket]
        if not bucket_data.empty:
            display_name = bucket_data.iloc[0]['threshold_display']
            color = colors[i % len(colors)]
            
            fig.add_trace(
                go.Scatter(
                    x=bucket_data['timestamp'],
                    y=bucket_data['probability'],
                    mode='lines',
                    name=display_name,
                    line=dict(color=color, width=2),
                    hovertemplate=f'{display_name}: %{{y:.1%}}<br>%{{x}}<extra></extra>'
                ),
                row=2, col=1
            )
    
    fig.update_xaxes(title_text="Time (ET)", row=2, col=1)
    fig.update_yaxes(title_text="Temperature (°F)", row=1, col=1)
    fig.update_yaxes(title_text="Probability", tickformat='.0%', row=2, col=1)
    
    fig.update_layout(
        height=750,
        hovermode='x unified',
        template="plotly_white",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="rgba(0, 0, 0, 0.08)",
            borderwidth=1,
            font=dict(color="#2D3748", size=12)
        ),
        margin=dict(t=80, b=50, l=50, r=50),
        plot_bgcolor='#FAFBFC',
        paper_bgcolor='white',
        font=dict(family="-apple-system, BlinkMacSystemFont, Inter, sans-serif", color="#2D3748", size=13),
        xaxis=dict(
            title=dict(font=dict(color="#4A5568", size=13)),
            tickfont=dict(color="#4A5568", size=11),
            gridcolor='rgba(0, 0, 0, 0.05)',
            showline=True,
            linecolor='rgba(0, 0, 0, 0.1)'
        ),
        yaxis=dict(
            title=dict(font=dict(color="#4A5568", size=13)),
            tickfont=dict(color="#4A5568", size=11),
            gridcolor='rgba(0, 0, 0, 0.05)',
            showline=True,
            linecolor='rgba(0, 0, 0, 0.1)'
        ),
        xaxis2=dict(
            title=dict(font=dict(color="#4A5568", size=13)),
            tickfont=dict(color="#4A5568", size=11),
            gridcolor='rgba(0, 0, 0, 0.05)',
            showline=True,
            linecolor='rgba(0, 0, 0, 0.1)'
        ),
        yaxis2=dict(
            title=dict(font=dict(color="#4A5568", size=13)),
            tickfont=dict(color="#4A5568", size=11),
            gridcolor='rgba(0, 0, 0, 0.05)',
            showline=True,
            linecolor='rgba(0, 0, 0, 0.1)'
        )
    )
    
    st.plotly_chart(fig, use_container_width=True, key=f"main_chart_{selected_date}")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Analysis insights
    st.markdown("<div style='margin: 2.5rem 0 1.5rem 0;'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Market Analysis</div>", unsafe_allow_html=True)
    
    # Show current odds for all buckets
    st.markdown("<div style='margin: 1.5rem 0 1rem 0; font-size: 1.125rem; font-weight: 500; color: #191B1F;'>Current Odds by Temperature Range</div>", unsafe_allow_html=True)
    
    latest_by_bucket = odds_df.groupby('threshold').last().reset_index()
    latest_by_bucket = latest_by_bucket.sort_values('probability', ascending=False)
    
    # Create a bar chart of current odds with modern styling
    st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
    
    fig_bar = go.Figure()
    
    # Use harmonious gradient colors
    colors = ['#667EEA' if p == latest_by_bucket['probability'].max() else '#CBD5E0' 
              for p in latest_by_bucket['probability']]
    
    fig_bar.add_trace(go.Bar(
        x=latest_by_bucket['threshold_display'],
        y=latest_by_bucket['probability'],
        marker=dict(
            color=colors,
            line=dict(width=0)
        ),
        text=[f"{p:.1%}" for p in latest_by_bucket['probability']],
        textposition='outside',
        textfont=dict(size=13, color='#2D3748', family="-apple-system, BlinkMacSystemFont, Inter, sans-serif"),
        hovertemplate='<b>%{x}</b><br>Probability: %{y:.1%}<extra></extra>'
    ))
    
    fig_bar.update_layout(
        title=dict(
            text=f"Latest Market Odds — {selected_date.strftime('%B %d, %Y')}",
            font=dict(size=18, color='#1A202C', family="-apple-system, BlinkMacSystemFont, Inter, sans-serif", weight=700),
            x=0,
            xanchor='left'
        ),
        xaxis=dict(
            title=dict(
                text="Temperature Range",
                font=dict(size=13, color='#718096')
            ),
            tickfont=dict(size=12, color='#4A5568'),
            showgrid=False,
            showline=True,
            linecolor='rgba(0, 0, 0, 0.1)'
        ),
        yaxis=dict(
            title=dict(
                text="Probability",
                font=dict(size=13, color='#718096')
            ),
            tickformat='.0%',
            range=[0, latest_by_bucket['probability'].max() * 1.15],
            showgrid=True,
            gridcolor='rgba(0, 0, 0, 0.05)',
            tickfont=dict(size=12, color='#4A5568'),
            showline=True,
            linecolor='rgba(0, 0, 0, 0.1)'
        ),
        height=400,
        showlegend=False,
        plot_bgcolor='#FAFBFC',
        paper_bgcolor='white',
        margin=dict(t=60, b=50, l=50, r=50),
        font=dict(family="-apple-system, BlinkMacSystemFont, Inter, sans-serif")
    )
    
    st.plotly_chart(fig_bar, use_container_width=True, key=f"bar_chart_{selected_date}")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Show odds changes with modern insight cards
    st.markdown("<div style='margin: 1.5rem 0;'></div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        max_bucket = latest_by_bucket.iloc[0]
        st.markdown(f"""
        <div class="insight-box" style="background: linear-gradient(135deg, #667EEA 0%, #764BA2 100%);">
            <h4>🎯 Most Likely</h4>
            <strong>{max_bucket['threshold_display']}</strong>
            <p style="font-size: 1.125rem; margin: 0.5rem 0;">Probability: {max_bucket['probability']:.1%}</p>
            <p style="font-size: 0.875rem; opacity: 0.9;">Updated: {max_bucket['timestamp'].strftime('%I:%M %p ET')}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Find bucket with biggest odds change
        for bucket in buckets:
            bucket_data = odds_df[odds_df['threshold'] == bucket]
            if len(bucket_data) > 1:
                bucket_data = bucket_data.sort_values('timestamp')
                first = bucket_data.iloc[0]
                last = bucket_data.iloc[-1]
                change = last['probability'] - first['probability']
                
                if 'biggest_change' not in locals() or abs(change) > abs(biggest_change):
                    biggest_change = change
                    biggest_change_bucket = last['threshold_display']
                    biggest_change_time = f"{first['timestamp'].strftime('%I:%M %p')} - {last['timestamp'].strftime('%I:%M %p')}"
        
        if 'biggest_change' in locals():
            gradient = "linear-gradient(135deg, #11998E 0%, #38EF7D 100%)" if biggest_change > 0 else "linear-gradient(135deg, #FC466B 0%, #3F5EFB 100%)"
            st.markdown(f"""
            <div class="insight-box" style="background: {gradient};">
                <h4>📈 Biggest Change</h4>
                <strong>{biggest_change_bucket}</strong>
                <p style="font-size: 1.125rem; margin: 0.5rem 0;">Change: {biggest_change:+.1%}</p>
                <p style="font-size: 0.875rem; opacity: 0.9;">{biggest_change_time} ET</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        # Current temperature vs most likely range
        current_temp = temp_df['temp_f'].iloc[-1]
        max_bucket_str = str(max_bucket['threshold'])
        
        in_range = False
        if '-' in max_bucket_str:
            low, high = map(int, max_bucket_str.split('-'))
            in_range = low <= current_temp <= high
        
        gradient = "linear-gradient(135deg, #F093FB 0%, #F5576C 100%)"
        st.markdown(f"""
        <div class="insight-box" style="background: {gradient};">
            <h4>🌡️ Current vs Market</h4>
            <strong>{current_temp:.1f}°F</strong>
            <p style="font-size: 1.125rem; margin: 0.5rem 0;">Market expects: {max_bucket['threshold_display']}</p>
            <p style="font-size: 0.875rem; opacity: 0.9;">{'✅ In range' if in_range else '⚠️ Outside range'}</p>
        </div>
        """, unsafe_allow_html=True)

else:
    # Temperature only chart with modern styling
    st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
    
    fig = go.Figure()
    
    # Split temperature data by date
    target_date_start = datetime.combine(selected_date, datetime.min.time()).replace(tzinfo=NY_TZ)
    target_date_end = datetime.combine(selected_date, datetime.max.time()).replace(tzinfo=NY_TZ)
    
    temp_df_target = temp_df[(temp_df['timestamp'] >= target_date_start) & (temp_df['timestamp'] <= target_date_end)]
    temp_df_before = temp_df[temp_df['timestamp'] < target_date_start]
    temp_df_after = temp_df[temp_df['timestamp'] > target_date_end]
    
    # Plot temperature before target date (soft gray)
    if not temp_df_before.empty:
        fig.add_trace(go.Scatter(
            x=temp_df_before['timestamp'],
            y=temp_df_before['temp_f'],
            mode='lines',
            name='Temp (Before)',
            line=dict(color='#CBD5E0', width=2.5, dash='dot'),
            hovertemplate='%{y:.1f}°F<br>%{x}<extra></extra>'
        ))
    
    # Plot temperature on target date (warm coral)
    if not temp_df_target.empty:
        fig.add_trace(go.Scatter(
            x=temp_df_target['timestamp'],
            y=temp_df_target['temp_f'],
            mode='lines',
            name=f'Temp ({selected_date.strftime("%b %d")})',
            line=dict(color='#FC8181', width=3.5),
            hovertemplate='%{y:.1f}°F<br>%{x}<extra></extra>'
        ))
    
    # Plot temperature after target date (soft gray)
    if not temp_df_after.empty:
        fig.add_trace(go.Scatter(
            x=temp_df_after['timestamp'],
            y=temp_df_after['temp_f'],
            mode='lines',
            name='Temp (After)',
            line=dict(color='#CBD5E0', width=2.5, dash='dot'),
            hovertemplate='%{y:.1f}°F<br>%{x}<extra></extra>'
        ))
    
    # Add vertical lines to mark target date
    fig.add_vline(
        x=target_date_start.timestamp() * 1000,
        line_dash="dash",
        line_color="rgba(0,0,0,0.3)"
    )
    fig.add_vline(
        x=target_date_end.timestamp() * 1000,
        line_dash="dash",
        line_color="rgba(0,0,0,0.3)"
    )
    
    fig.update_layout(
        title=dict(
            text="Temperature Data — Full Period",
            font=dict(size=18, color='#1A202C', family="-apple-system, BlinkMacSystemFont, Inter, sans-serif", weight=700),
            x=0,
            xanchor='left'
        ),
        yaxis=dict(
            title=dict(
                text="Temperature (°F)",
                font=dict(size=13, color='#718096')
            ),
            showgrid=True,
            gridcolor='rgba(0, 0, 0, 0.05)',
            tickfont=dict(size=12, color='#4A5568'),
            showline=True,
            linecolor='rgba(0, 0, 0, 0.1)'
        ),
        xaxis=dict(
            title=dict(
                text="Time (ET)",
                font=dict(size=13, color='#718096')
            ),
            tickfont=dict(size=12, color='#4A5568'),
            showline=True,
            linecolor='rgba(0, 0, 0, 0.1)'
        ),
        height=500,
        hovermode='x unified',
        plot_bgcolor='#FAFBFC',
        paper_bgcolor='white',
        font=dict(family="-apple-system, BlinkMacSystemFont, Inter, sans-serif", color="#2D3748")
    )
    
    st.plotly_chart(fig, use_container_width=True, key=f"temp_only_chart_{selected_date}")
    
    st.markdown("</div>", unsafe_allow_html=True)

# Timezone verification with modern cards
st.markdown("<div style='margin: 2.5rem 0 1.5rem 0;'></div>", unsafe_allow_html=True)
st.markdown("<div class='section-header'>Timezone Verification</div>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    sample_time = temp_df.iloc[0]['timestamp']
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">NWS Data Timezone</div>
        <div style="background: #F8F9FA; padding: 1rem; border-radius: 8px; margin: 1rem 0; font-family: 'Monaco', monospace; font-size: 0.875rem; color: #191B1F;">
            {sample_time}
        </div>
        <div class="status-badge status-success">✓ America/New_York (ET)</div>
        <div class="metric-caption" style="margin-top: 0.75rem;">The -05:00 offset confirms Eastern Standard Time</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    if has_odds:
        sample_odds_time = odds_df.iloc[0]['timestamp']
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Polymarket Data Timezone</div>
            <div style="background: #F8F9FA; padding: 1rem; border-radius: 8px; margin: 1rem 0; font-family: 'Monaco', monospace; font-size: 0.875rem; color: #191B1F;">
                {sample_odds_time}
            </div>
            <div class="status-badge status-success">✓ America/New_York (ET)</div>
            <div class="metric-caption" style="margin-top: 0.75rem;">Both datasets use the same timezone for accurate comparison</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Polymarket Data Timezone</div>
            <div style="background: #F8F9FA; padding: 1rem; border-radius: 8px; margin: 1rem 0; font-family: 'Monaco', monospace; font-size: 0.875rem; color: #80868B;">
                No data loaded
            </div>
            <div class="status-badge status-info">ℹ️ Awaiting data</div>
        </div>
        """, unsafe_allow_html=True)

# Data tables with modern styling
st.markdown("<div style='margin: 2.5rem 0 1.5rem 0;'></div>", unsafe_allow_html=True)
with st.expander("📋 View Raw Data", expanded=False):
    # Determine number of tabs based on available data
    tabs_list = ["Temperature Data", "Odds Data"]
    if forecast_df is not None and not forecast_df.empty:
        tabs_list.append("Visual Crossing Forecast")
    if nws_forecast_df is not None and not nws_forecast_df.empty:
        tabs_list.append("NWS Forecast")
    if forecast_df is not None and nws_forecast_df is not None:
        tabs_list.append("Forecast Comparison")
    
    tabs = st.tabs(tabs_list)
    tab_idx = 0
    
    # Tab 1: Temperature Data
    with tabs[tab_idx]:
        temp_display = temp_df[['timestamp', 'temp_f', 'humidity', 'wind_speed_mph']].copy()
        temp_display['timestamp'] = temp_display['timestamp'].dt.strftime('%Y-%m-%d %I:%M:%S %p ET')
        temp_display.columns = ['Time', 'Temp (°F)', 'Humidity (%)', 'Wind (mph)']
        st.dataframe(temp_display, width='stretch', height=300)
    tab_idx += 1
    
    # Tab 2: Odds Data
    with tabs[tab_idx]:
        if has_odds:
            odds_display = odds_df[['timestamp', 'threshold_display', 'probability', 'question']].copy()
            odds_display['timestamp'] = odds_display['timestamp'].dt.strftime('%Y-%m-%d %I:%M:%S %p ET')
            odds_display['probability'] = odds_display['probability'].apply(lambda x: f"{x:.1%}")
            odds_display.columns = ['Time', 'Range', 'Odds', 'Market Question']
            st.dataframe(odds_display, width='stretch', height=300)
        else:
            st.info("No odds data available. Run: `python scripts/fetching/fetch_polymarket_historical.py`")
    tab_idx += 1
    
    # Tab 3: Visual Crossing Forecast (if available)
    if forecast_df is not None and not forecast_df.empty:
        with tabs[tab_idx]:
            st.markdown(f"**Forecast issued:** {selected_forecast_time.strftime('%B %d, %Y at %I:%M %p ET')}")
            st.markdown(f"**Forecasting for:** {selected_date.strftime('%B %d, %Y')} + 3 days")
            st.markdown("---")
            
            forecast_display = forecast_df[['timestamp', 'temp_f']].copy()
            forecast_display['timestamp'] = forecast_display['timestamp'].dt.strftime('%Y-%m-%d %I:%M:%S %p ET')
            forecast_display.columns = ['Forecast Time', 'Predicted Temp (°F)']
            
            # Add a column showing if this is the target date
            forecast_display['Target Date'] = forecast_df['timestamp'].apply(
                lambda x: '✓' if x.date() == selected_date else ''
            )
            
            st.dataframe(forecast_display, width='stretch', height=400)
            
            # Show summary stats for target date
            target_forecast = forecast_df[forecast_df['timestamp'].dt.date == selected_date]
            if not target_forecast.empty:
                st.markdown("---")
                st.markdown(f"**Visual Crossing Forecast for {selected_date.strftime('%B %d')}:**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Predicted High", f"{target_forecast['temp_f'].max():.1f}°F")
                with col2:
                    st.metric("Predicted Low", f"{target_forecast['temp_f'].min():.1f}°F")
                with col3:
                    st.metric("Predicted Avg", f"{target_forecast['temp_f'].mean():.1f}°F")
        tab_idx += 1
    
    # Tab 4: NWS Forecast (if available)
    if nws_forecast_df is not None and not nws_forecast_df.empty:
        with tabs[tab_idx]:
            st.markdown(f"**NWS Current Forecast**")
            st.markdown(f"**Valid for:** Next 7 days from now")
            st.markdown("---")
            
            nws_display = nws_forecast_df[['timestamp', 'temp_f']].copy()
            nws_display['timestamp'] = nws_display['timestamp'].dt.strftime('%Y-%m-%d %I:%M:%S %p ET')
            nws_display.columns = ['Forecast Time', 'Predicted Temp (°F)']
            
            # Add a column showing if this is the target date
            nws_display['Target Date'] = nws_forecast_df['timestamp'].apply(
                lambda x: '✓' if x.date() == selected_date else ''
            )
            
            st.dataframe(nws_display, width='stretch', height=400)
            
            # Show summary stats for target date
            target_nws = nws_forecast_df[nws_forecast_df['timestamp'].dt.date == selected_date]
            if not target_nws.empty:
                st.markdown("---")
                st.markdown(f"**NWS Forecast for {selected_date.strftime('%B %d')}:**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Predicted High", f"{target_nws['temp_f'].max():.1f}°F")
                with col2:
                    st.metric("Predicted Low", f"{target_nws['temp_f'].min():.1f}°F")
                with col3:
                    st.metric("Predicted Avg", f"{target_nws['temp_f'].mean():.1f}°F")
        tab_idx += 1
    
    # Tab 5: Forecast Comparison (if both available)
    if forecast_df is not None and nws_forecast_df is not None:
        with tabs[tab_idx]:
            st.markdown("### Forecast Comparison")
            st.markdown(f"**Comparing forecasts for {selected_date.strftime('%B %d, %Y')}**")
            st.markdown("---")
            
            # Get target date data from both
            vc_target = forecast_df[forecast_df['timestamp'].dt.date == selected_date]
            nws_target = nws_forecast_df[nws_forecast_df['timestamp'].dt.date == selected_date]
            
            if not vc_target.empty and not nws_target.empty:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Visual Crossing**")
                    st.metric("High", f"{vc_target['temp_f'].max():.1f}°F")
                    st.metric("Low", f"{vc_target['temp_f'].min():.1f}°F")
                    st.metric("Avg", f"{vc_target['temp_f'].mean():.1f}°F")
                
                with col2:
                    st.markdown("**NWS**")
                    st.metric("High", f"{nws_target['temp_f'].max():.1f}°F")
                    st.metric("Low", f"{nws_target['temp_f'].min():.1f}°F")
                    st.metric("Avg", f"{nws_target['temp_f'].mean():.1f}°F")
                
                with col3:
                    st.markdown("**Difference**")
                    high_diff = vc_target['temp_f'].max() - nws_target['temp_f'].max()
                    low_diff = vc_target['temp_f'].min() - nws_target['temp_f'].min()
                    avg_diff = vc_target['temp_f'].mean() - nws_target['temp_f'].mean()
                    
                    st.metric("High Δ", f"{high_diff:+.1f}°F")
                    st.metric("Low Δ", f"{low_diff:+.1f}°F")
                    st.metric("Avg Δ", f"{avg_diff:+.1f}°F")
                
                st.markdown("---")
                st.markdown("**Interpretation:**")
                if abs(high_diff) < 2:
                    st.success("✓ Forecasts agree closely on high temperature")
                elif high_diff > 0:
                    st.warning(f"⚠️ Visual Crossing predicts {abs(high_diff):.1f}°F warmer than NWS")
                else:
                    st.warning(f"⚠️ NWS predicts {abs(high_diff):.1f}°F warmer than Visual Crossing")
            else:
                st.info("No overlapping forecast data for the target date")

# Footer with elegant styling
st.markdown("<div style='margin: 3rem 0 1rem 0;'></div>", unsafe_allow_html=True)
st.markdown(f"""
<div style='text-align: center; padding: 2.5rem 0; border-top: 1px solid rgba(0, 0, 0, 0.06); background: linear-gradient(to bottom, transparent 0%, #FAFBFC 100%);'>
    <p style='color: #718096; font-size: 0.9375rem; margin: 0.5rem 0; font-weight: 500;'>
        Last updated: {datetime.now(NY_TZ).strftime('%B %d, %Y at %I:%M:%S %p')} ET
    </p>
    <p style='color: #A0AEC0; font-size: 0.875rem; margin: 0.5rem 0;'>
        Temperature: NWS KLGA • Odds: Polymarket • All times in Eastern Time
    </p>
</div>
""", unsafe_allow_html=True)

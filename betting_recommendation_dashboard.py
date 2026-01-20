"""
Betting Recommendation Dashboard

Real-time betting recommendations based on Open-Meteo forecasts and backtest strategy.
Shows which markets to bet on and how much based on your bankroll.

Run: streamlit run betting_recommendation_dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, date, timezone
import sys
import os
import requests
import json
import re
import numpy as np
from scipy import stats

sys.path.insert(0, '.')

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

NY_TZ = ZoneInfo("America/New_York")

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Get Visual Crossing API key - try Streamlit secrets first, then .env
try:
    VISUAL_CROSSING_API_KEY = st.secrets.get("VISUAL_CROSSING_API_KEY")
except:
    VISUAL_CROSSING_API_KEY = os.getenv("VISUAL_CROSSING_API_KEY")

# Debug: Show if key is loaded (first/last 4 chars only)
if VISUAL_CROSSING_API_KEY and VISUAL_CROSSING_API_KEY != "your_visual_crossing_key_here":
    masked_key = f"{VISUAL_CROSSING_API_KEY[:4]}...{VISUAL_CROSSING_API_KEY[-4:]}"
    print(f"Visual Crossing API key loaded: {masked_key}")
else:
    print("Visual Crossing API key not found in secrets or .env")

# Page config
st.set_page_config(
    page_title="Betting Recommendations - KLGA",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styling
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(to bottom, #FAFBFC 0%, #F5F7FA 100%) !important;
    }
    
    .metric-card {
        background: linear-gradient(to bottom right, #FFFFFF 0%, #F7FAFC 100%);
        padding: 1.75rem;
        border-radius: 12px;
        border: 1px solid rgba(0, 0, 0, 0.06);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }
    
    .bet-card {
        background: linear-gradient(135deg, #48BB78 0%, #38A169 100%);
        padding: 1.75rem;
        border-radius: 12px;
        color: white !important;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(72, 187, 120, 0.25);
    }
    
    .bet-card h4, .bet-card p, .bet-card strong {
        color: white !important;
    }
    
    .no-bet-card {
        background: linear-gradient(135deg, #FC8181 0%, #F56565 100%);
        padding: 1.75rem;
        border-radius: 12px;
        color: white !important;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(245, 101, 101, 0.25);
    }
    
    .no-bet-card h4, .no-bet-card p, .no-bet-card strong {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Load error model
@st.cache_data
def load_error_model(lead_time='1d'):
    """Load error distribution model for specified lead time"""
    try:
        with open('data/processed/error_distribution_analysis.json', 'r') as f:
            model = json.load(f)
        
        if 'by_lead_time' in model and lead_time in model['by_lead_time']:
            return model['by_lead_time'][lead_time]
        else:
            return model.get('overall', model)
    except:
        return None

# Fetch Open-Meteo forecast
@st.cache_data(ttl=300)
def fetch_openmeteo_forecast(target_date):
    """Fetch Open-Meteo forecast for target date"""
    url = "https://api.open-meteo.com/v1/forecast"
    
    # LaGuardia Airport coordinates
    lat = 40.7769
    lon = -73.8740
    
    # Calculate days ahead
    days_ahead = (target_date - date.today()).days
    
    # Open-Meteo forecast API starts from tomorrow (doesn't include today)
    if days_ahead < 1 or days_ahead > 16:
        return None
    
    # Get forecast - always get at least 3 days to ensure we have data
    params = {
        'latitude': lat,
        'longitude': lon,
        'hourly': 'temperature_2m',
        'temperature_unit': 'fahrenheit',
        'timezone': 'America/New_York',
        'forecast_days': max(3, days_ahead + 1)  # Get enough days
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
        
        records = []
        for time_str, temp in zip(times, temps):
            if temp is None:
                continue
            
            # Parse timestamp - Open-Meteo returns ISO format strings
            timestamp = datetime.fromisoformat(time_str).replace(tzinfo=NY_TZ)
            
            records.append({
                'timestamp': timestamp,
                'temp_f': round(temp, 1),
                'date': timestamp.date()
            })
        
        if not records:
            # Debug: show what dates we got
            if times:
                first_date = datetime.fromisoformat(times[0]).date()
                last_date = datetime.fromisoformat(times[-1]).date()
                st.warning(f"No data for {target_date}. Got data from {first_date} to {last_date}")
            return None
        
        df = pd.DataFrame(records)
        return df
        
    except Exception as e:
        st.error(f"Error fetching Open-Meteo forecast: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None
        return None

# Fetch NWS forecast
@st.cache_data(ttl=300)
def fetch_nws_forecast():
    """Fetch NWS forecast for KLGA"""
    try:
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
                
                df['date'] = df['timestamp'].dt.date
                return df[['timestamp', 'temp_f', 'date']].copy()
        
        return None
        
    except Exception as e:
        return None

# Fetch Visual Crossing forecast
@st.cache_data(ttl=3600)
def fetch_visual_crossing_forecast(target_date):
    """Fetch Visual Crossing forecast - includes current day if available"""
    # Use the global API key
    if not VISUAL_CROSSING_API_KEY:
        st.sidebar.warning("⚠️ VISUAL_CROSSING_API_KEY not found in .env file")
        return None
    
    if VISUAL_CROSSING_API_KEY == "your_visual_crossing_key_here":
        st.sidebar.warning("⚠️ VISUAL_CROSSING_API_KEY is set to placeholder value")
        return None
    
    location = "LaGuardia Airport,NY,US"
    
    # Get data starting from today (not target_date) to include current day
    today = date.today()
    start_date = today.isoformat()
    # Get enough days to cover target_date + buffer
    days_ahead = (target_date - today).days
    end_date = (today + timedelta(days=max(3, days_ahead + 1))).isoformat()
    
    url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{location}/{start_date}/{end_date}"
    
    params = {
        'key': VISUAL_CROSSING_API_KEY,
        'unitGroup': 'us',
        'include': 'hours',
        'contentType': 'json',
        'elements': 'datetime,temp'
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code == 401:
            st.sidebar.error("❌ Visual Crossing API key is invalid")
            return None
        elif response.status_code == 429:
            st.sidebar.warning("⚠️ Visual Crossing API rate limit exceeded")
            return None
        elif response.status_code != 200:
            st.sidebar.warning(f"⚠️ Visual Crossing API error: {response.status_code}")
            return None
        
        response.raise_for_status()
        data = response.json()
        
        if 'days' not in data:
            return None
        
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
                            'date': timestamp.date()
                        })
        
        if not records:
            return None
        
        df = pd.DataFrame(records)
        df = df.sort_values('timestamp')
        
        st.sidebar.success(f"✓ Visual Crossing: {len(df)} hours from {df['date'].min()} to {df['date'].max()}")
        
        return df
        
    except Exception as e:
        st.sidebar.error(f"❌ Visual Crossing error: {str(e)}")
        return None

# Fetch actual temperature readings from NWS
@st.cache_data(ttl=60)
def fetch_actual_temperatures(target_date):
    """Fetch actual temperature observations from NWS (includes current day up to now)"""
    try:
        from src.data.weather_scraper import WeatherScraper
        
        scraper = WeatherScraper(station_id="KLGA")
        
        # Get data from 3 days before to now (in UTC for API)
        now_utc = datetime.now(timezone.utc)
        start_dt = now_utc - timedelta(days=3)
        
        # Fetch in 24-hour chunks
        all_dfs = []
        current_start = start_dt
        
        while current_start < now_utc:
            current_end = min(current_start + timedelta(hours=24), now_utc)
            
            try:
                df_chunk = scraper.fetch_raw_observations(current_start, current_end)
                if not df_chunk.empty:
                    all_dfs.append(df_chunk)
            except Exception as e:
                pass
            
            current_start = current_end
        
        if not all_dfs:
            return None
        
        # Combine all chunks
        df = pd.concat(all_dfs, ignore_index=False)
        df = df.sort_index()
        df = df[~df.index.duplicated(keep='first')]
        
        # Convert to NY timezone
        df.index = df.index.tz_convert(NY_TZ)
        df = df.reset_index()
        df = df.rename(columns={'index': 'timestamp'})
        
        # Add date column in NY timezone
        df['date'] = df['timestamp'].dt.date
        
        # Debug info
        if not df.empty:
            date_range = f"{df['date'].min()} to {df['date'].max()}"
            st.sidebar.success(f"✓ NWS Actual: {len(df)} readings from {date_range}")
        
        return df[['timestamp', 'temp_f', 'date']].copy()
        
    except Exception as e:
        st.sidebar.warning(f"⚠️ Could not fetch NWS actual temps: {str(e)}")
        return None

# Fetch Polymarket odds
@st.cache_data(ttl=60)
def fetch_polymarket_odds(target_date):
    """Fetch current Polymarket odds for target date"""
    month = target_date.strftime('%B').lower()
    day = target_date.day
    slug = f"highest-temperature-in-nyc-on-{month}-{day}"
    
    api_url = f"https://gamma-api.polymarket.com/events?slug={slug}"
    headers = {
        'User-Agent': 'Mozilla/5.0',
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
        
        records = []
        
        for market in markets:
            question = market.get('question', '')
            
            # Parse threshold
            range_match = re.search(r'between (\d+)-(\d+)°F', question)
            if range_match:
                low = int(range_match.group(1))
                high = int(range_match.group(2))
                threshold = f"{low}-{high}"
                threshold_value = (low + high) / 2
                threshold_type = 'range'
            elif 'or below' in question.lower() or 'or lower' in question.lower():
                temp_match = re.search(r'(\d+)°F', question)
                if temp_match:
                    temp = int(temp_match.group(1))
                    threshold = f"≤{temp}"
                    threshold_value = temp
                    threshold_type = 'below'
                else:
                    continue
            elif 'or higher' in question.lower() or 'or above' in question.lower():
                temp_match = re.search(r'(\d+)°F', question)
                if temp_match:
                    temp = int(temp_match.group(1))
                    threshold = f"≥{temp}"
                    threshold_value = temp
                    threshold_type = 'above'
                else:
                    continue
            else:
                continue
            
            # Get current price
            outcome_prices = market.get('outcomePrices', [])
            
            # Handle JSON string or array
            if isinstance(outcome_prices, str):
                try:
                    outcome_prices = json.loads(outcome_prices)
                except:
                    outcome_prices = []
            
            if outcome_prices and len(outcome_prices) > 0:
                try:
                    yes_price = float(outcome_prices[0])
                except (ValueError, TypeError):
                    continue
            else:
                continue
            
            # Get volume
            volume = float(market.get('volume', 0))
            
            records.append({
                'threshold': threshold,
                'threshold_value': threshold_value,
                'threshold_type': threshold_type,
                'market_prob': yes_price,
                'volume': volume,
                'question': question
            })
        
        if not records:
            return None
        
        df = pd.DataFrame(records)
        return df
        
    except Exception as e:
        st.error(f"Error fetching Polymarket odds: {e}")
        return None

# Calculate model probability
def calculate_model_probability(forecast_temp, threshold_value, threshold_type, error_model):
    """Calculate model probability using error distribution"""
    mean_error = error_model['mean']
    std_error = error_model['std']
    
    adjusted_forecast = forecast_temp - mean_error
    
    if threshold_type == 'above':
        z_score = (threshold_value - adjusted_forecast) / std_error
        return 1 - stats.norm.cdf(z_score)
    
    elif threshold_type == 'below':
        z_score = (threshold_value - adjusted_forecast) / std_error
        return stats.norm.cdf(z_score)
    
    elif threshold_type == 'range':
        # For range like "32-33", we need the actual bounds
        # Polymarket rounds to whole degrees, so:
        # - 32 means [31.5, 32.5)
        # - 33 means [32.5, 33.5)
        # - Combined "32-33" means [31.5, 33.5)
        # threshold_value is the midpoint (e.g., 32.5 for "32-33")
        # So we need to go ±1.0 from midpoint to get full range
        low = threshold_value - 1.0
        high = threshold_value + 1.0
        
        z_low = (low - adjusted_forecast) / std_error
        z_high = (high - adjusted_forecast) / std_error
        
        return stats.norm.cdf(z_high) - stats.norm.cdf(z_low)
    
    return 0.0

def should_consider_no_bet(forecasted_max, threshold_value, threshold_type, min_distance=2):
    """
    Determine if we should consider betting NO on a threshold.
    
    Logic: If forecast is 50°F, we might bet NO on 46°F or 54°F 
    because we're confident the temp will be different.
    """
    if threshold_type != 'range':
        # Only consider NO bets on range markets for now
        return False
    
    # Bet NO if the threshold is significantly away from our forecast
    distance = abs(forecasted_max - threshold_value)
    
    return distance >= min_distance

def calculate_no_bet_probability(forecast_temp, threshold_value, threshold_type, error_model):
    """
    Calculate probability that a NO bet wins (i.e., actual temp does NOT meet threshold).
    NO bet wins when YES bet loses.
    """
    yes_prob = calculate_model_probability(forecast_temp, threshold_value, threshold_type, error_model)
    return 1 - yes_prob

# Generate betting recommendations
def generate_recommendations(forecast_max, odds_df, error_model, bankroll, min_edge=0.05, min_ev=0.05, 
                            min_volume=100, max_bet_vs_volume=0.10, kelly_fraction=0.25, max_bet_pct=0.10,
                            enable_no_bets=True, no_bet_min_distance=2):
    """
    Generate betting recommendations based on forecast and odds.
    
    Args:
        forecast_max: Forecasted maximum temperature for the betting day
        odds_df: DataFrame with market odds
        error_model: Error distribution model
        bankroll: Available bankroll
        ... (other parameters)
    
    Returns:
        DataFrame with recommendations
    """
    
    if forecast_max is None or odds_df is None or error_model is None:
        return None
    
    recommendations = []
    
    for _, market in odds_df.iterrows():
        threshold_value = market['threshold_value']
        threshold_type = market['threshold_type']
        market_prob = market['market_prob']
        volume = market['volume']
        
        # === YES BET LOGIC ===
        # Calculate model probability for YES
        model_prob_yes = calculate_model_probability(
            forecast_max, threshold_value, threshold_type, error_model
        )
        
        # Calculate edge and EV for YES
        edge_yes = model_prob_yes - market_prob
        
        if market_prob > 0 and market_prob < 1:
            payout_multiplier_yes = 1 / market_prob
            ev_yes = (model_prob_yes * payout_multiplier_yes) - 1
        else:
            ev_yes = 0
        
        # Check if YES bet meets criteria
        should_bet_yes = (
            (edge_yes >= min_edge) and 
            (ev_yes >= min_ev) and 
            (market_prob >= 0.05) and
            (market_prob <= 0.95) and
            (volume >= min_volume)
        )
        
        # Calculate bet size for YES
        bet_size_yes = 0
        liquidity_constrained_yes = False
        
        if should_bet_yes and market_prob > 0 and market_prob < 1:
            # Kelly criterion
            b = (1 / market_prob) - 1
            kelly = (b * model_prob_yes - (1 - model_prob_yes)) / b
            kelly = max(0, min(kelly, 1))
            bet_size_yes = kelly * kelly_fraction * bankroll
            bet_size_yes = min(bet_size_yes, bankroll * max_bet_pct)
            
            # Apply liquidity constraint
            max_bet_from_volume = volume * max_bet_vs_volume
            if bet_size_yes > max_bet_from_volume:
                bet_size_yes = max_bet_from_volume
                liquidity_constrained_yes = True
        
        recommendations.append({
            'threshold': market['threshold'],
            'threshold_value': threshold_value,
            'threshold_type': threshold_type,
            'bet_side': 'YES',
            'market_prob': market_prob,
            'model_prob': model_prob_yes,
            'edge': edge_yes,
            'ev': ev_yes,
            'volume': volume,
            'should_bet': should_bet_yes,
            'bet_size': bet_size_yes,
            'liquidity_constrained': liquidity_constrained_yes,
            'forecasted_max': forecast_max,
            'distance_from_forecast': abs(forecast_max - threshold_value)
        })
        
        # === NO BET LOGIC ===
        if enable_no_bets and should_consider_no_bet(forecast_max, threshold_value, threshold_type, no_bet_min_distance):
            # Calculate model probability for NO (probability that YES loses)
            model_prob_no = calculate_no_bet_probability(
                forecast_max, threshold_value, threshold_type, error_model
            )
            
            # Market probability for NO is (1 - yes_probability)
            market_prob_no = 1 - market_prob
            
            # Calculate edge and EV for NO
            edge_no = model_prob_no - market_prob_no
            
            if market_prob_no > 0 and market_prob_no < 1:
                payout_multiplier_no = 1 / market_prob_no
                ev_no = (model_prob_no * payout_multiplier_no) - 1
            else:
                ev_no = 0
            
            # Check if NO bet meets criteria
            should_bet_no = (
                (edge_no >= min_edge) and 
                (ev_no >= min_ev) and 
                (market_prob_no >= 0.05) and
                (market_prob_no <= 0.95) and
                (volume >= min_volume)
            )
            
            # Calculate bet size for NO
            bet_size_no = 0
            liquidity_constrained_no = False
            
            if should_bet_no and market_prob_no > 0 and market_prob_no < 1:
                b = (1 / market_prob_no) - 1
                kelly = (b * model_prob_no - (1 - model_prob_no)) / b
                kelly = max(0, min(kelly, 1))
                bet_size_no = kelly * kelly_fraction * bankroll
                bet_size_no = min(bet_size_no, bankroll * max_bet_pct)
                
                max_bet_from_volume = volume * max_bet_vs_volume
                if bet_size_no > max_bet_from_volume:
                    bet_size_no = max_bet_from_volume
                    liquidity_constrained_no = True
            
            recommendations.append({
                'threshold': market['threshold'],
                'threshold_value': threshold_value,
                'threshold_type': threshold_type,
                'bet_side': 'NO',
                'market_prob': market_prob_no,
                'model_prob': model_prob_no,
                'edge': edge_no,
                'ev': ev_no,
                'volume': volume,
                'should_bet': should_bet_no,
                'bet_size': bet_size_no,
                'liquidity_constrained': liquidity_constrained_no,
                'forecasted_max': forecast_max,
                'distance_from_forecast': abs(forecast_max - threshold_value)
            })
    
    df = pd.DataFrame(recommendations)
    df = df.sort_values('edge', ascending=False)
    
    return df

# Title
st.markdown("""
<div style='margin-bottom: 2rem; padding: 2rem 0 1.5rem 0; border-bottom: 1px solid rgba(0, 0, 0, 0.06);'>
    <h1 style='font-size: 2.75rem; font-weight: 700; background: linear-gradient(135deg, #667EEA 0%, #764BA2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0;'>
        💰 Betting Recommendations
    </h1>
    <p style='font-size: 1.125rem; color: #718096; margin-top: 0.75rem;'>
        Real-time betting analysis using Open-Meteo forecasts and proven strategy
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar controls
st.sidebar.markdown("## ⚙️ Settings")

# Date selector
today = date.today()
tomorrow = today + timedelta(days=1)

selected_date = st.sidebar.date_input(
    "Market Date",
    value=tomorrow,
    min_value=today,  # Can select today
    max_value=today + timedelta(days=16),
    help="Select which date's market to analyze (today uses Visual Crossing, future dates use Open-Meteo)"
)

# Bankroll input
bankroll = st.sidebar.number_input(
    "Your Bankroll ($)",
    min_value=100,
    max_value=100000,
    value=1000,
    step=100,
    help="How much money you have available to bet"
)

# Strategy parameters
st.sidebar.markdown("### Strategy Parameters")

min_edge = st.sidebar.slider(
    "Minimum Edge (%)",
    min_value=0,
    max_value=20,
    value=5,
    step=1,
    help="Minimum advantage required (model prob - market prob)"
) / 100

kelly_fraction = st.sidebar.slider(
    "Kelly Fraction (%)",
    min_value=10,
    max_value=50,
    value=25,
    step=5,
    help="Fraction of Kelly criterion to use (lower = more conservative)"
) / 100

max_bet_pct = st.sidebar.slider(
    "Max Bet (% of bankroll)",
    min_value=1,
    max_value=20,
    value=10,
    step=1,
    help="Maximum bet size as percentage of bankroll"
) / 100

# NO betting strategy
st.sidebar.markdown("### NO Betting Strategy")

enable_no_bets = st.sidebar.checkbox(
    "Enable NO Bets",
    value=True,
    help="Bet NO on ranges far from your forecast (e.g., if forecast is 50°F, bet NO on 44°F)"
)

no_bet_min_distance = st.sidebar.slider(
    "NO Bet Min Distance (°F)",
    min_value=1,
    max_value=5,
    value=2,
    step=1,
    help="Minimum distance from forecast to bet NO (2°F recommended)",
    disabled=not enable_no_bets
) if enable_no_bets else 2

# Lead time selection
lead_time_option = st.sidebar.radio(
    "Forecast Lead Time",
    options=["Same-day (0d)", "1-day ahead (1d)"],
    index=0,
    help="Which error model to use"
)
lead_time = '0d' if 'Same-day' in lead_time_option else '1d'

# Forecast source selection
st.sidebar.markdown("### Forecast Source")

# Determine available sources based on selected date
is_today = (selected_date == date.today())
available_sources = []

if not is_today:
    available_sources.append("Open-Meteo")

available_sources.extend(["NWS", "Visual Crossing", "Average of All"])

# Set default based on date
default_index = 0
if is_today and "Visual Crossing" in available_sources:
    default_index = available_sources.index("Visual Crossing")

forecast_source = st.sidebar.radio(
    "Which forecast to use for max temp",
    options=available_sources,
    index=default_index,
    help="Select which forecast source to use for betting recommendations. Open-Meteo not available for current day."
)

st.sidebar.markdown("---")
refresh_btn = st.sidebar.button("🔄 Refresh Data")

# Load error model
error_model = load_error_model(lead_time)

if error_model is None:
    st.error("❌ Error model not found. Run: `python scripts/analysis/check_error_distribution.py`")
    st.stop()

# Fetch data
with st.spinner("Loading forecast and market data..."):
    # Determine if we're looking at today
    is_today = (selected_date == date.today())
    
    # Show current time for debugging
    current_time_et = datetime.now(NY_TZ)
    st.sidebar.info(f"🕐 Current time: {current_time_et.strftime('%b %d, %Y %I:%M %p ET')}")
    
    # For today, prioritize Visual Crossing (has current day data)
    # For future dates, use Open-Meteo
    if is_today:
        st.sidebar.info("📅 Today's market - using Visual Crossing for current day forecast")
        forecast_df = None  # Open-Meteo doesn't have today
    else:
        forecast_df = fetch_openmeteo_forecast(selected_date)
    
    odds_df = fetch_polymarket_odds(selected_date)
    nws_forecast_df = fetch_nws_forecast()
    vc_forecast_df = fetch_visual_crossing_forecast(selected_date)
    actual_temps_df = fetch_actual_temperatures(selected_date)

# Check if data is available
if forecast_df is None and vc_forecast_df is None and nws_forecast_df is None:
    st.error(f"❌ No forecast data available for {selected_date}")
    days_ahead = (selected_date - date.today()).days
    
    if days_ahead < 0:
        st.info("💡 Cannot get forecast for past dates. Select today or a future date.")
    elif days_ahead == 0:
        st.info("💡 For today's forecast, Visual Crossing API key is required.")
        st.info("📝 Add VISUAL_CROSSING_API_KEY to your .env file or Streamlit secrets")
    elif days_ahead > 16:
        st.info("💡 Open-Meteo only provides forecasts up to 16 days ahead.")
    else:
        st.info("💡 Try refreshing or selecting a different date.")
    st.stop()

if odds_df is None:
    st.warning(f"⚠️ No Polymarket market found for {selected_date}")
    st.info("💡 Market may not be open yet. Markets typically open 2 days before resolution.")
    st.info(f"Try selecting {(date.today() + timedelta(days=1)).strftime('%B %d')} or {(date.today() + timedelta(days=2)).strftime('%B %d')}")
    st.stop()

# Calculate forecasted max based on selected source
forecasted_max_betting_day = None
is_today = (selected_date == date.today())

# For today, we might have partial actual data - use it if available
if is_today and actual_temps_df is not None:
    actual_today = actual_temps_df[actual_temps_df['date'] == selected_date]
    if not actual_today.empty:
        current_max = actual_today['temp_f'].max()
        st.sidebar.info(f"📊 Current max today: {current_max:.1f}°F (so far)")

if forecast_source == "Open-Meteo" and forecast_df is not None:
    betting_day_data = forecast_df[forecast_df['date'] == selected_date]
    if not betting_day_data.empty:
        forecasted_max_betting_day = betting_day_data['temp_f'].max()

elif forecast_source == "NWS" and nws_forecast_df is not None:
    betting_day_data = nws_forecast_df[nws_forecast_df['date'] == selected_date]
    if not betting_day_data.empty:
        forecasted_max_betting_day = betting_day_data['temp_f'].max()

elif forecast_source == "Visual Crossing" and vc_forecast_df is not None:
    betting_day_data = vc_forecast_df[vc_forecast_df['date'] == selected_date]
    if not betting_day_data.empty:
        forecasted_max_betting_day = betting_day_data['temp_f'].max()

elif forecast_source == "Average of All":
    # Calculate average of all available forecasts
    max_temps = []
    
    if forecast_df is not None:
        betting_day_data = forecast_df[forecast_df['date'] == selected_date]
        if not betting_day_data.empty:
            max_temps.append(betting_day_data['temp_f'].max())
    
    if nws_forecast_df is not None:
        betting_day_data = nws_forecast_df[nws_forecast_df['date'] == selected_date]
        if not betting_day_data.empty:
            max_temps.append(betting_day_data['temp_f'].max())
    
    if vc_forecast_df is not None:
        betting_day_data = vc_forecast_df[vc_forecast_df['date'] == selected_date]
        if not betting_day_data.empty:
            max_temps.append(betting_day_data['temp_f'].max())
    
    if max_temps:
        forecasted_max_betting_day = sum(max_temps) / len(max_temps)

# Generate recommendations
recommendations = generate_recommendations(
    forecasted_max_betting_day, odds_df, error_model, bankroll,
    min_edge=min_edge, min_ev=0.05, min_volume=100,
    max_bet_vs_volume=0.10, kelly_fraction=kelly_fraction, max_bet_pct=max_bet_pct,
    enable_no_bets=enable_no_bets, no_bet_min_distance=no_bet_min_distance
)

# Display error model info
st.markdown("### 📊 Model Information")

# Show special notice for today's market
is_today = (selected_date == date.today())
if is_today:
    st.info("""
    📅 **Today's Market** - Using real-time data where available. 
    - Actual temps shown up to current time
    - Forecast used for remaining hours
    - Consider current max when evaluating bets
    """)

col1, col2, col3, col4 = st.columns(4)

with col1:
    if forecasted_max_betting_day is not None:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 0.8rem; color: #718096; text-transform: uppercase; letter-spacing: 0.8px;">Forecast Max ({selected_date.strftime('%b %d')})</div>
            <div style="font-size: 2rem; font-weight: 700; color: #667EEA; margin: 0.5rem 0;">{forecasted_max_betting_day:.1f}°F</div>
            <div style="font-size: 0.875rem; color: #A0AEC0;">{forecast_source}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 0.8rem; color: #718096; text-transform: uppercase; letter-spacing: 0.8px;">Forecast Max</div>
            <div style="font-size: 2rem; font-weight: 700; color: #CBD5E0; margin: 0.5rem 0;">—</div>
            <div style="font-size: 0.875rem; color: #A0AEC0;">{forecast_source} not available</div>
        </div>
        """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div style="font-size: 0.8rem; color: #718096; text-transform: uppercase; letter-spacing: 0.8px;">Model Bias</div>
        <div style="font-size: 2rem; font-weight: 700; color: #667EEA; margin: 0.5rem 0;">{error_model['mean']:+.2f}°F</div>
        <div style="font-size: 0.875rem; color: #A0AEC0;">{lead_time_option}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div style="font-size: 0.8rem; color: #718096; text-transform: uppercase; letter-spacing: 0.8px;">Std Dev</div>
        <div style="font-size: 2rem; font-weight: 700; color: #667EEA; margin: 0.5rem 0;">{error_model['std']:.2f}°F</div>
        <div style="font-size: 0.875rem; color: #A0AEC0;">MAE: {error_model['mae']:.2f}°F</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div style="font-size: 0.8rem; color: #718096; text-transform: uppercase; letter-spacing: 0.8px;">Your Bankroll</div>
        <div style="font-size: 2rem; font-weight: 700; color: #667EEA; margin: 0.5rem 0;">${bankroll:,.0f}</div>
        <div style="font-size: 0.875rem; color: #A0AEC0;">Max bet: ${bankroll * max_bet_pct:.0f}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Forecast Comparison Section
st.markdown("### 🌡️ Temperature Forecasts Comparison")

# Calculate forecasted max for ONLY the target betting day from each source
openmeteo_max = None
nws_max = None
vc_max = None
actual_max_so_far = None

# Check if we have actual data for today
is_today = (selected_date == date.today())
if is_today and actual_temps_df is not None:
    actual_today = actual_temps_df[actual_temps_df['date'] == selected_date]
    if not actual_today.empty:
        actual_max_so_far = actual_today['temp_f'].max()
        # Debug: show time range of actual data
        first_time = actual_today['timestamp'].min().strftime('%I:%M %p')
        last_time = actual_today['timestamp'].max().strftime('%I:%M %p')
        st.sidebar.info(f"📊 Today's actual data: {first_time} to {last_time} ET ({len(actual_today)} readings)")
    else:
        st.sidebar.warning(f"⚠️ No actual data found for {selected_date} yet")

# Get forecasted max for the betting day only
if forecast_df is not None:
    # Filter to only the target betting day
    target_day_data = forecast_df[forecast_df['date'] == selected_date]
    if not target_day_data.empty:
        openmeteo_max = target_day_data['temp_f'].max()

if nws_forecast_df is not None:
    # Filter to only the target betting day
    target_day_data = nws_forecast_df[nws_forecast_df['date'] == selected_date]
    if not target_day_data.empty:
        nws_max = target_day_data['temp_f'].max()

if vc_forecast_df is not None:
    # Filter to only the target betting day
    target_day_data = vc_forecast_df[vc_forecast_df['date'] == selected_date]
    if not target_day_data.empty:
        vc_max = target_day_data['temp_f'].max()

# Show info about selected forecast source
if forecast_source == "Average of All":
    available_sources = []
    if openmeteo_max is not None:
        available_sources.append(f"Open-Meteo ({openmeteo_max:.1f}°F)")
    if nws_max is not None:
        available_sources.append(f"NWS ({nws_max:.1f}°F)")
    if vc_max is not None:
        available_sources.append(f"Visual Crossing ({vc_max:.1f}°F)")
    
    if available_sources:
        st.info(f"📊 Using average of: {', '.join(available_sources)} = **{forecasted_max_betting_day:.1f}°F**")
else:
    st.info(f"📊 Using **{forecast_source}** forecast for betting recommendations (highlighted with green border below)")

# Show current max if today
if is_today and actual_max_so_far is not None:
    st.success(f"🌡️ **Current max today (so far): {actual_max_so_far:.1f}°F** - Forecast predicts max of {forecasted_max_betting_day:.1f}°F by end of day")

# Display forecast max cards with highlighting for selected source
num_cols = 4 if (is_today and actual_max_so_far is not None) else 3
cols = st.columns(num_cols)

# Show actual max first if today
col_idx = 0
if is_today and actual_max_so_far is not None:
    with cols[col_idx]:
        st.markdown(f"""
        <div class="metric-card" style="border: 3px solid #E53E3E; box-shadow: 0 4px 16px rgba(229, 62, 62, 0.3);">
            <div style="font-size: 0.8rem; color: #718096; text-transform: uppercase; letter-spacing: 0.8px;">Actual Max (So Far)</div>
            <div style="font-size: 2rem; font-weight: 700; color: #E53E3E; margin: 0.5rem 0;">{actual_max_so_far:.1f}°F</div>
            <div style="font-size: 0.875rem; color: #A0AEC0;">As of {datetime.now(NY_TZ).strftime('%I:%M %p ET')}</div>
        </div>
        """, unsafe_allow_html=True)
    col_idx += 1

with cols[col_idx]:
    selected_style = "border: 3px solid #48BB78; box-shadow: 0 4px 16px rgba(72, 187, 120, 0.3);" if forecast_source == "Open-Meteo" or (forecast_source == "Average of All" and openmeteo_max is not None) else ""
    if openmeteo_max is not None:
        st.markdown(f"""
        <div class="metric-card" style="{selected_style}">
            <div style="font-size: 0.8rem; color: #718096; text-transform: uppercase; letter-spacing: 0.8px;">Open-Meteo Max {'✓' if forecast_source == 'Open-Meteo' else ''}</div>
            <div style="font-size: 2rem; font-weight: 700; color: #FC8181; margin: 0.5rem 0;">{openmeteo_max:.1f}°F</div>
            <div style="font-size: 0.875rem; color: #A0AEC0;">For {selected_date.strftime('%b %d')}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        unavailable_reason = "Not available for today" if is_today else "Not available"
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 0.8rem; color: #718096; text-transform: uppercase; letter-spacing: 0.8px;">Open-Meteo Max</div>
            <div style="font-size: 2rem; font-weight: 700; color: #CBD5E0; margin: 0.5rem 0;">—</div>
            <div style="font-size: 0.875rem; color: #A0AEC0;">{unavailable_reason}</div>
        </div>
        """, unsafe_allow_html=True)
col_idx += 1

with cols[col_idx]:
    selected_style = "border: 3px solid #48BB78; box-shadow: 0 4px 16px rgba(72, 187, 120, 0.3);" if forecast_source == "NWS" or (forecast_source == "Average of All" and nws_max is not None) else ""
    if nws_max is not None:
        st.markdown(f"""
        <div class="metric-card" style="{selected_style}">
            <div style="font-size: 0.8rem; color: #718096; text-transform: uppercase; letter-spacing: 0.8px;">NWS Max {'✓' if forecast_source == 'NWS' else ''}</div>
            <div style="font-size: 2rem; font-weight: 700; color: #4FACFE; margin: 0.5rem 0;">{nws_max:.1f}°F</div>
            <div style="font-size: 0.875rem; color: #A0AEC0;">For {selected_date.strftime('%b %d')}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 0.8rem; color: #718096; text-transform: uppercase; letter-spacing: 0.8px;">NWS Max</div>
            <div style="font-size: 2rem; font-weight: 700; color: #CBD5E0; margin: 0.5rem 0;">—</div>
            <div style="font-size: 0.875rem; color: #A0AEC0;">Not available</div>
        </div>
        """, unsafe_allow_html=True)
col_idx += 1

with cols[col_idx]:
    selected_style = "border: 3px solid #48BB78; box-shadow: 0 4px 16px rgba(72, 187, 120, 0.3);" if forecast_source == "Visual Crossing" or (forecast_source == "Average of All" and vc_max is not None) else ""
    if vc_max is not None:
        st.markdown(f"""
        <div class="metric-card" style="{selected_style}">
            <div style="font-size: 0.8rem; color: #718096; text-transform: uppercase; letter-spacing: 0.8px;">Visual Crossing Max {'✓' if forecast_source == 'Visual Crossing' else ''}</div>
            <div style="font-size: 2rem; font-weight: 700; color: #764BA2; margin: 0.5rem 0;">{vc_max:.1f}°F</div>
            <div style="font-size: 0.875rem; color: #A0AEC0;">For {selected_date.strftime('%b %d')}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 0.8rem; color: #718096; text-transform: uppercase; letter-spacing: 0.8px;">Visual Crossing Max</div>
            <div style="font-size: 2rem; font-weight: 700; color: #CBD5E0; margin: 0.5rem 0;">—</div>
            <div style="font-size: 0.875rem; color: #A0AEC0;">Not available</div>
        </div>
        """, unsafe_allow_html=True)

# Create forecast comparison chart
if forecast_df is not None or nws_forecast_df is not None or vc_forecast_df is not None or actual_temps_df is not None:
    fig_forecast = go.Figure()
    
    # Define dynamic window based on selected date
    now = datetime.now(NY_TZ)
    window_start = now - timedelta(hours=24)  # 24 hours ago
    
    # Calculate end of target date
    target_date_end_dt = datetime.combine(selected_date + timedelta(days=1), datetime.min.time()).replace(tzinfo=NY_TZ)
    
    # Extend window to at least cover the target date + 12 hours buffer
    min_window_end = target_date_end_dt + timedelta(hours=12)
    window_end = max(now + timedelta(hours=48), min_window_end)  # At least 48 hours or until target date + buffer
    
    # Define target date boundaries at midnight ET
    target_date_start = datetime.combine(selected_date, datetime.min.time()).replace(tzinfo=NY_TZ)
    target_date_end = datetime.combine(selected_date + timedelta(days=1), datetime.min.time()).replace(tzinfo=NY_TZ)
    
    # Add actual temperature readings (if available)
    if actual_temps_df is not None:
        # Filter to window (from 24h ago to now)
        actual_window = actual_temps_df[
            (actual_temps_df['timestamp'] >= window_start) & 
            (actual_temps_df['timestamp'] <= now)
        ]
        
        if not actual_window.empty:
            # Debug: show what dates we have
            unique_dates = actual_window['date'].unique()
            st.sidebar.info(f"📊 Actual temps available for: {', '.join([str(d) for d in sorted(unique_dates)])}")
            
            # Split by date relative to target
            actual_before = actual_window[actual_window['date'] < selected_date]
            actual_target = actual_window[actual_window['date'] == selected_date]
            
            # Debug: show counts
            if not actual_target.empty:
                st.sidebar.success(f"✓ {len(actual_target)} actual readings for {selected_date}")
            
            # Plot actual temps before target date (gray)
            if not actual_before.empty:
                fig_forecast.add_trace(go.Scatter(
                    x=actual_before['timestamp'],
                    y=actual_before['temp_f'],
                    mode='lines',
                    name='Actual (Past)',
                    line=dict(color='#A0AEC0', width=2.5),
                    hovertemplate='Actual: %{y:.1f}°F<br>%{x}<extra></extra>'
                ))
            
            # Plot actual temps on target date (bold red)
            if not actual_target.empty:
                fig_forecast.add_trace(go.Scatter(
                    x=actual_target['timestamp'],
                    y=actual_target['temp_f'],
                    mode='lines',
                    name=f'Actual ({selected_date.strftime("%b %d")})',
                    line=dict(color='#E53E3E', width=3.5),
                    hovertemplate='Actual: %{y:.1f}°F<br>%{x}<extra></extra>'
                ))
        else:
            st.sidebar.info(f"ℹ️ No actual temps in window ({window_start.strftime('%b %d %I:%M%p')} to now)")
    
    # Add Open-Meteo forecast (filter to window)
    if forecast_df is not None:
        forecast_window = forecast_df[
            (forecast_df['timestamp'] >= now) & 
            (forecast_df['timestamp'] <= window_end)
        ]
        if not forecast_window.empty:
            fig_forecast.add_trace(go.Scatter(
                x=forecast_window['timestamp'],
                y=forecast_window['temp_f'],
                mode='lines+markers',
                name='Open-Meteo Forecast',
                line=dict(color='#FC8181', width=3),
                marker=dict(size=5),
                hovertemplate='Open-Meteo: %{y:.1f}°F<br>%{x}<extra></extra>'
            ))
    
    # Add NWS forecast (filter to window)
    if nws_forecast_df is not None:
        forecast_window = nws_forecast_df[
            (nws_forecast_df['timestamp'] >= now) & 
            (nws_forecast_df['timestamp'] <= window_end)
        ]
        if not forecast_window.empty:
            fig_forecast.add_trace(go.Scatter(
                x=forecast_window['timestamp'],
                y=forecast_window['temp_f'],
                mode='lines+markers',
                name='NWS Forecast',
                line=dict(color='#4FACFE', width=2.5, dash='dot'),
                marker=dict(size=4),
                hovertemplate='NWS: %{y:.1f}°F<br>%{x}<extra></extra>'
            ))
    
    # Add Visual Crossing forecast (filter to window)
    if vc_forecast_df is not None:
        forecast_window = vc_forecast_df[
            (vc_forecast_df['timestamp'] >= now) & 
            (vc_forecast_df['timestamp'] <= window_end)
        ]
        if not forecast_window.empty:
            fig_forecast.add_trace(go.Scatter(
                x=forecast_window['timestamp'],
                y=forecast_window['temp_f'],
                mode='lines+markers',
                name='Visual Crossing Forecast',
                line=dict(color='#764BA2', width=2.5, dash='dash'),
                marker=dict(size=4),
                hovertemplate='Visual Crossing: %{y:.1f}°F<br>%{x}<extra></extra>'
            ))
    
    # Add vertical line for current time using shapes
    fig_forecast.add_shape(
        type="line",
        x0=now, x1=now,
        y0=0, y1=1,
        yref="paper",
        line=dict(color="rgba(0,0,0,0.5)", width=2, dash="solid")
    )
    fig_forecast.add_annotation(
        x=now, y=1, yref="paper",
        text="Now", showarrow=False,
        yshift=10, font=dict(size=10)
    )
    
    # Add vertical lines to mark target date boundaries (if in window)
    if target_date_start >= window_start and target_date_start <= window_end:
        fig_forecast.add_shape(
            type="line",
            x0=target_date_start, x1=target_date_start,
            y0=0, y1=1,
            yref="paper",
            line=dict(color="rgba(102, 126, 234, 0.5)", width=2, dash="dash")
        )
        fig_forecast.add_annotation(
            x=target_date_start, y=1, yref="paper",
            text=f"{selected_date.strftime('%b %d')} starts", showarrow=False,
            yshift=10, font=dict(size=10)
        )
    
    if target_date_end >= window_start and target_date_end <= window_end:
        fig_forecast.add_shape(
            type="line",
            x0=target_date_end, x1=target_date_end,
            y0=0, y1=1,
            yref="paper",
            line=dict(color="rgba(102, 126, 234, 0.5)", width=2, dash="dash")
        )
        fig_forecast.add_annotation(
            x=target_date_end, y=1, yref="paper",
            text=f"{selected_date.strftime('%b %d')} ends", showarrow=False,
            yshift=10, font=dict(size=10)
        )
    
    # Calculate hours shown for title
    hours_shown = int((window_end - window_start).total_seconds() / 3600)
    
    fig_forecast.update_layout(
        title=f"{hours_shown}-Hour Temperature Window - {selected_date.strftime('%B %d, %Y')}",
        xaxis_title="Time (ET)",
        yaxis_title="Temperature (°F)",
        height=500,
        hovermode='x unified',
        template="plotly_white",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis=dict(
            range=[window_start, window_end]
        )
    )
    
    st.plotly_chart(fig_forecast, use_container_width=True)
else:
    st.info("No forecast data available for comparison chart")

st.markdown("---")

# Display recommendations
st.markdown(f"### 🎯 Betting Recommendations for {selected_date.strftime('%B %d, %Y')}")

bets_to_place = recommendations[recommendations['should_bet'] == True]
yes_bets = bets_to_place[bets_to_place['bet_side'] == 'YES']
no_bets = bets_to_place[bets_to_place['bet_side'] == 'NO']

if len(bets_to_place) == 0:
    st.markdown("""
    <div class="no-bet-card">
        <h4 style="margin: 0 0 0.5rem 0;">❌ No Bets Recommended</h4>
        <p style="margin: 0;">No markets meet the minimum edge criteria. Either the market is fairly priced or there's insufficient edge to justify a bet.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    # Summary stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Bets", len(bets_to_place))
    with col2:
        st.metric("YES Bets", len(yes_bets), delta=f"{len(yes_bets)/len(bets_to_place)*100:.0f}%" if len(bets_to_place) > 0 else "0%")
    with col3:
        st.metric("NO Bets", len(no_bets), delta=f"{len(no_bets)/len(bets_to_place)*100:.0f}%" if len(bets_to_place) > 0 else "0%")
    
    # Show YES bets first
    if len(yes_bets) > 0:
        st.markdown("#### ✅ YES Bets (Bet that temperature WILL be in range)")
        for idx, bet in yes_bets.iterrows():
            liquidity_note = " [LIQUIDITY CAPPED]" if bet['liquidity_constrained'] else ""
            
            st.markdown(f"""
            <div class="bet-card">
                <h4 style="margin: 0 0 1rem 0; font-size: 1.25rem;">🎲 BET YES: {bet['threshold']}</h4>
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem; margin-bottom: 1rem;">
                    <div>
                        <div style="font-size: 0.75rem; opacity: 0.9; text-transform: uppercase; letter-spacing: 0.8px;">Bet Size{liquidity_note}</div>
                        <div style="font-size: 1.75rem; font-weight: 700;">${bet['bet_size']:.2f}</div>
                    </div>
                    <div>
                        <div style="font-size: 0.75rem; opacity: 0.9; text-transform: uppercase; letter-spacing: 0.8px;">Edge</div>
                        <div style="font-size: 1.75rem; font-weight: 700;">{bet['edge']:+.1%}</div>
                    </div>
                    <div>
                        <div style="font-size: 0.75rem; opacity: 0.9; text-transform: uppercase; letter-spacing: 0.8px;">Model Probability</div>
                        <div style="font-size: 1.25rem; font-weight: 600;">{bet['model_prob']:.1%}</div>
                    </div>
                    <div>
                        <div style="font-size: 0.75rem; opacity: 0.9; text-transform: uppercase; letter-spacing: 0.8px;">Market Probability</div>
                        <div style="font-size: 1.25rem; font-weight: 600;">{bet['market_prob']:.1%}</div>
                    </div>
                    <div>
                        <div style="font-size: 0.75rem; opacity: 0.9; text-transform: uppercase; letter-spacing: 0.8px;">Expected Value</div>
                        <div style="font-size: 1.25rem; font-weight: 600;">{bet['ev']:+.1%}</div>
                    </div>
                    <div>
                        <div style="font-size: 0.75rem; opacity: 0.9; text-transform: uppercase; letter-spacing: 0.8px;">Market Volume</div>
                        <div style="font-size: 1.25rem; font-weight: 600;">${bet['volume']:,.0f}</div>
                    </div>
                </div>
                <div style="background: rgba(255,255,255,0.15); padding: 0.75rem; border-radius: 6px; font-size: 0.875rem;">
                    <strong>Potential Profit:</strong> ${bet['bet_size'] * ((1/bet['market_prob']) - 1):.2f} if win | Loss: ${bet['bet_size']:.2f} if lose
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Show NO bets
    if len(no_bets) > 0:
        st.markdown("#### 🚫 NO Bets (Bet that temperature will NOT be in range)")
        
        # Add explanation for NO bets
        st.info(f"""
        💡 **Why bet NO?** Your forecast is {forecasted_max_betting_day:.1f}°F. These ranges are {no_bet_min_distance}+ degrees away, 
        so you're confident the temperature won't land there. NO bets typically have 95%+ win rate!
        """)
        
        for idx, bet in no_bets.iterrows():
            liquidity_note = " [LIQUIDITY CAPPED]" if bet['liquidity_constrained'] else ""
            distance = bet['distance_from_forecast']
            direction = "below" if bet['threshold_value'] < forecasted_max_betting_day else "above"
            
            st.markdown(f"""
            <div class="no-bet-card">
                <h4 style="margin: 0 0 0.5rem 0; font-size: 1.25rem; color: white !important;">🚫 BET NO: {bet['threshold']}</h4>
                <p style="margin: 0 0 1rem 0; font-size: 0.875rem; opacity: 0.9; color: white !important;">
                    {distance:.1f}°F {direction} your forecast of {forecasted_max_betting_day:.1f}°F
                </p>
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem; margin-bottom: 1rem;">
                    <div>
                        <div style="font-size: 0.75rem; opacity: 0.9; text-transform: uppercase; letter-spacing: 0.8px; color: white !important;">Bet Size{liquidity_note}</div>
                        <div style="font-size: 1.75rem; font-weight: 700; color: white !important;">${bet['bet_size']:.2f}</div>
                    </div>
                    <div>
                        <div style="font-size: 0.75rem; opacity: 0.9; text-transform: uppercase; letter-spacing: 0.8px; color: white !important;">Edge</div>
                        <div style="font-size: 1.75rem; font-weight: 700; color: white !important;">{bet['edge']:+.1%}</div>
                    </div>
                    <div>
                        <div style="font-size: 0.75rem; opacity: 0.9; text-transform: uppercase; letter-spacing: 0.8px; color: white !important;">Model Prob (NO wins)</div>
                        <div style="font-size: 1.25rem; font-weight: 600; color: white !important;">{bet['model_prob']:.1%}</div>
                    </div>
                    <div>
                        <div style="font-size: 0.75rem; opacity: 0.9; text-transform: uppercase; letter-spacing: 0.8px; color: white !important;">Market Prob (NO wins)</div>
                        <div style="font-size: 1.25rem; font-weight: 600; color: white !important;">{bet['market_prob']:.1%}</div>
                    </div>
                    <div>
                        <div style="font-size: 0.75rem; opacity: 0.9; text-transform: uppercase; letter-spacing: 0.8px; color: white !important;">Expected Value</div>
                        <div style="font-size: 1.25rem; font-weight: 600; color: white !important;">{bet['ev']:+.1%}</div>
                    </div>
                    <div>
                        <div style="font-size: 0.75rem; opacity: 0.9; text-transform: uppercase; letter-spacing: 0.8px; color: white !important;">Market Volume</div>
                        <div style="font-size: 1.25rem; font-weight: 600; color: white !important;">${bet['volume']:,.0f}</div>
                    </div>
                </div>
                <div style="background: rgba(255,255,255,0.15); padding: 0.75rem; border-radius: 6px; font-size: 0.875rem; color: white !important;">
                    <strong style="color: white !important;">Potential Profit:</strong> ${bet['bet_size'] * ((1/bet['market_prob']) - 1):.2f} if win | Loss: ${bet['bet_size']:.2f} if lose
                </div>
            </div>
            """, unsafe_allow_html=True)

# Show all markets for comparison
st.markdown("---")
st.markdown("### 📊 All Markets")

# Sort recommendations by temperature range (lowest to highest)
def sort_key(threshold):
    """Sort key for temperature thresholds"""
    threshold_str = str(threshold)
    if threshold_str.startswith('≤'):
        # Below thresholds come first, sorted by value
        return (0, float(threshold_str[1:]))
    elif '-' in threshold_str:
        # Range thresholds in the middle, sorted by lower bound
        low = float(threshold_str.split('-')[0])
        return (1, low)
    elif threshold_str.startswith('≥'):
        # Above thresholds come last, sorted by value
        return (2, float(threshold_str[1:]))
    else:
        return (1, 999)  # Fallback

# Filter to only YES bets for the chart (to avoid duplication)
# Each temperature range should only appear once
recommendations_yes_only = recommendations[recommendations['bet_side'] == 'YES'].copy()
recommendations_yes_only['sort_key'] = recommendations_yes_only['threshold'].apply(sort_key)
recommendations_yes_only = recommendations_yes_only.sort_values('sort_key')

# Create comparison chart showing YES probabilities only
fig = go.Figure()

fig.add_trace(go.Bar(
    x=recommendations_yes_only['threshold'],
    y=recommendations_yes_only['model_prob'],
    name='Model Probability (YES)',
    marker_color='#667EEA',
    hovertemplate='%{x}<br>Model: %{y:.1%}<extra></extra>'
))

fig.add_trace(go.Bar(
    x=recommendations_yes_only['threshold'],
    y=recommendations_yes_only['market_prob'],
    name='Market Probability (YES)',
    marker_color='#CBD5E0',
    hovertemplate='%{x}<br>Market: %{y:.1%}<extra></extra>'
))

fig.update_layout(
    title="Model vs Market Probabilities (YES Bets Only)",
    xaxis_title="Temperature Range (Lowest to Highest)",
    yaxis_title="Probability that temp WILL be in range",
    yaxis_tickformat='.0%',
    barmode='group',
    height=400,
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)

# Add a second chart showing recommended bets only (both YES and NO)
if len(bets_to_place) > 0:
    st.markdown("### 🎯 Recommended Bets Visualization")
    
    # Sort recommended bets by threshold
    bets_sorted = bets_to_place.copy()
    bets_sorted['sort_key'] = bets_sorted['threshold'].apply(sort_key)
    bets_sorted = bets_sorted.sort_values('sort_key')
    
    # Create chart showing edge for recommended bets
    fig_bets = go.Figure()
    
    # Color code by bet side
    colors = ['#48BB78' if side == 'YES' else '#F56565' for side in bets_sorted['bet_side']]
    
    fig_bets.add_trace(go.Bar(
        x=[f"{row['bet_side']}: {row['threshold']}" for _, row in bets_sorted.iterrows()],
        y=bets_sorted['edge'],
        name='Edge',
        marker_color=colors,
        text=[f"{edge:+.1%}" for edge in bets_sorted['edge']],
        textposition='outside',
        hovertemplate='%{x}<br>Edge: %{y:+.1%}<br>Bet Size: $%{customdata:.2f}<extra></extra>',
        customdata=bets_sorted['bet_size'],
        cliponaxis=False  # Don't clip text labels
    ))
    
    # Calculate y-axis range to fit labels
    max_edge = bets_sorted['edge'].max()
    y_range_max = max_edge * 1.15  # Add 15% padding for labels
    
    fig_bets.update_layout(
        title="Edge by Recommended Bet (Green=YES, Red=NO)",
        xaxis_title="Bet",
        yaxis_title="Edge (Model Prob - Market Prob)",
        yaxis_tickformat='+.0%',
        yaxis_range=[0, y_range_max],  # Set range to fit labels
        height=450,
        template="plotly_white",
        showlegend=False,
        margin=dict(t=80, b=60, l=60, r=40)  # Add more top margin for labels
    )
    
    st.plotly_chart(fig_bets, use_container_width=True)

# Data table
st.markdown("### 📋 Detailed Analysis")

# Create tabs for different views
tab1, tab2 = st.tabs(["📊 All Opportunities", "✅ Recommended Bets Only"])

with tab1:
    st.markdown("**All temperature ranges with both YES and NO betting opportunities**")
    
    # Sort all recommendations
    recommendations_sorted = recommendations.copy()
    recommendations_sorted['sort_key'] = recommendations_sorted['threshold'].apply(sort_key)
    recommendations_sorted = recommendations_sorted.sort_values('sort_key')
    
    # Prepare display dataframe
    display_df_all = recommendations_sorted[['bet_side', 'threshold', 'model_prob', 'market_prob', 'edge', 'ev', 'volume', 'should_bet', 'bet_size', 'distance_from_forecast']].copy()
    display_df_all.columns = ['Side', 'Range', 'Model Prob', 'Market Prob', 'Edge', 'EV', 'Volume', 'Bet?', 'Bet Size', 'Distance']
    
    # Format columns
    display_df_all['Model Prob'] = display_df_all['Model Prob'].apply(lambda x: f"{x:.1%}")
    display_df_all['Market Prob'] = display_df_all['Market Prob'].apply(lambda x: f"{x:.1%}")
    display_df_all['Edge'] = display_df_all['Edge'].apply(lambda x: f"{x:+.1%}")
    display_df_all['EV'] = display_df_all['EV'].apply(lambda x: f"{x:+.1%}")
    display_df_all['Volume'] = display_df_all['Volume'].apply(lambda x: f"${x:,.0f}")
    display_df_all['Bet?'] = display_df_all['Bet?'].apply(lambda x: "✅ BET" if x else "—")
    display_df_all['Bet Size'] = display_df_all['Bet Size'].apply(lambda x: f"${x:.2f}" if x > 0 else "—")
    display_df_all['Distance'] = display_df_all['Distance'].apply(lambda x: f"{x:.1f}°F")
    
    # Apply styling to highlight recommended bets
    def highlight_recommended(row):
        if row['Bet?'] == '✅ BET':
            if row['Side'] == 'YES':
                return ['background-color: rgba(72, 187, 120, 0.15)'] * len(row)  # Green tint for YES
            else:
                return ['background-color: rgba(245, 101, 101, 0.15)'] * len(row)  # Light red tint for NO
        return [''] * len(row)
    
    styled_df = display_df_all.style.apply(highlight_recommended, axis=1)
    st.dataframe(styled_df, use_container_width=True, height=500)
    
    st.caption("💡 Highlighted rows are recommended bets: Green = YES bets, Light Red = NO bets")

with tab2:
    st.markdown("**Only bets that meet your criteria (ready to place)**")
    if len(bets_to_place) > 0:
        # Sort recommended bets
        bets_display_sorted = bets_to_place.copy()
        bets_display_sorted['sort_key'] = bets_display_sorted['threshold'].apply(sort_key)
        bets_display_sorted = bets_display_sorted.sort_values('sort_key')
        
        bets_display = bets_display_sorted[['bet_side', 'threshold', 'model_prob', 'market_prob', 'edge', 'ev', 'volume', 'bet_size', 'distance_from_forecast']].copy()
        bets_display.columns = ['Side', 'Range', 'Model Prob', 'Market Prob', 'Edge', 'EV', 'Volume', 'Bet Size', 'Distance']
        bets_display['Model Prob'] = bets_display['Model Prob'].apply(lambda x: f"{x:.1%}")
        bets_display['Market Prob'] = bets_display['Market Prob'].apply(lambda x: f"{x:.1%}")
        bets_display['Edge'] = bets_display['Edge'].apply(lambda x: f"{x:+.1%}")
        bets_display['EV'] = bets_display['EV'].apply(lambda x: f"{x:+.1%}")
        bets_display['Volume'] = bets_display['Volume'].apply(lambda x: f"${x:,.0f}")
        bets_display['Bet Size'] = bets_display['Bet Size'].apply(lambda x: f"${x:.2f}")
        bets_display['Distance'] = bets_display['Distance'].apply(lambda x: f"{x:.1f}°F")
        
        # Apply styling
        def highlight_bet_side(row):
            if row['Side'] == 'YES':
                return ['background-color: rgba(72, 187, 120, 0.2)'] * len(row)  # Green for YES
            else:
                return ['background-color: rgba(245, 101, 101, 0.2)'] * len(row)  # Light red for NO
        
        styled_bets = bets_display.style.apply(highlight_bet_side, axis=1)
        st.dataframe(styled_bets, use_container_width=True, height=400)
        
        # Summary stats
        total_capital = bets_display_sorted['bet_size'].sum()
        st.success(f"💰 **Total Capital to Deploy**: ${total_capital:.2f} across {len(bets_to_place)} bets")
    else:
        st.info("No bets meet your criteria. Try adjusting the minimum edge or other parameters.")


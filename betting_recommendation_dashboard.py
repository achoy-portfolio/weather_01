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

# Get Visual Crossing API key
VISUAL_CROSSING_API_KEY = os.getenv("VISUAL_CROSSING_API_KEY")

# Debug: Show if key is loaded (first/last 4 chars only)
if VISUAL_CROSSING_API_KEY and VISUAL_CROSSING_API_KEY != "your_visual_crossing_key_here":
    masked_key = f"{VISUAL_CROSSING_API_KEY[:4]}...{VISUAL_CROSSING_API_KEY[-4:]}"
    print(f"Visual Crossing API key loaded: {masked_key}")
else:
    print("Visual Crossing API key not found in .env")

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
        background: linear-gradient(135deg, #667EEA 0%, #764BA2 100%);
        padding: 1.75rem;
        border-radius: 12px;
        color: white !important;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(102, 126, 234, 0.25);
    }
    
    .bet-card h4, .bet-card p, .bet-card strong {
        color: white !important;
    }
    
    .no-bet-card {
        background: linear-gradient(135deg, #CBD5E0 0%, #A0AEC0 100%);
        padding: 1.75rem;
        border-radius: 12px;
        color: white !important;
        margin: 1rem 0;
    }
    
    .no-bet-card h4, .no-bet-card p {
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
    """Fetch actual temperature observations from NWS (up to 7 days past)"""
    try:
        from src.data.weather_scraper import WeatherScraper
        
        scraper = WeatherScraper(station_id="KLGA")
        
        # Get data from 3 days before to now
        now = datetime.now(timezone.utc)
        start_dt = now - timedelta(days=3)
        
        # Fetch in 24-hour chunks
        all_dfs = []
        current_start = start_dt
        
        while current_start < now:
            current_end = min(current_start + timedelta(hours=24), now)
            
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
        
        df.index = df.index.tz_convert(NY_TZ)
        df = df.reset_index()
        df = df.rename(columns={'index': 'timestamp'})
        
        df['date'] = df['timestamp'].dt.date
        
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

# Generate betting recommendations
def generate_recommendations(forecast_df, odds_df, error_model, target_date, bankroll, min_edge=0.05, min_ev=0.05, 
                            min_volume=100, max_bet_vs_volume=0.10, kelly_fraction=0.25, max_bet_pct=0.05):
    """Generate betting recommendations based on forecast and odds"""
    
    if forecast_df is None or odds_df is None or error_model is None:
        return None
    
    # Get forecasted max temperature for ONLY the betting day (target_date)
    betting_day_forecast = forecast_df[forecast_df['date'] == target_date]
    if betting_day_forecast.empty:
        return None
    
    forecasted_max = betting_day_forecast['temp_f'].max()
    
    recommendations = []
    
    for _, market in odds_df.iterrows():
        threshold_value = market['threshold_value']
        threshold_type = market['threshold_type']
        market_prob = market['market_prob']
        volume = market['volume']
        
        # Calculate model probability
        model_prob = calculate_model_probability(
            forecasted_max, threshold_value, threshold_type, error_model
        )
        
        # Calculate edge and EV
        edge = model_prob - market_prob
        
        if market_prob > 0 and market_prob < 1:
            payout_multiplier = 1 / market_prob
            ev = (model_prob * payout_multiplier) - 1
        else:
            ev = 0
        
        # Check if bet meets criteria
        should_bet = (
            (edge >= min_edge) and 
            (ev >= min_ev) and 
            (market_prob >= 0.05) and
            (market_prob <= 0.95) and
            (volume >= min_volume)
        )
        
        # Calculate bet size
        bet_size = 0
        liquidity_constrained = False
        
        if should_bet and market_prob > 0 and market_prob < 1:
            # Kelly criterion
            b = (1 / market_prob) - 1
            kelly = (b * model_prob - (1 - model_prob)) / b
            kelly = max(0, min(kelly, 1))
            bet_size = kelly * kelly_fraction * bankroll
            bet_size = min(bet_size, bankroll * max_bet_pct)
            
            # Apply liquidity constraint
            max_bet_from_volume = volume * max_bet_vs_volume
            if bet_size > max_bet_from_volume:
                bet_size = max_bet_from_volume
                liquidity_constrained = True
        
        recommendations.append({
            'threshold': market['threshold'],
            'threshold_value': threshold_value,
            'threshold_type': threshold_type,
            'market_prob': market_prob,
            'model_prob': model_prob,
            'edge': edge,
            'ev': ev,
            'volume': volume,
            'should_bet': should_bet,
            'bet_size': bet_size,
            'liquidity_constrained': liquidity_constrained,
            'forecasted_max': forecasted_max
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
    min_value=tomorrow,  # Can't select today since forecast starts from tomorrow
    max_value=today + timedelta(days=16),  # Open-Meteo supports up to 16 days
    help="Select which date's market to analyze (forecast available from tomorrow onwards)"
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
    max_value=10,
    value=5,
    step=1,
    help="Maximum bet size as percentage of bankroll"
) / 100

# Lead time selection
lead_time_option = st.sidebar.radio(
    "Forecast Lead Time",
    options=["Same-day (0d)", "1-day ahead (1d)"],
    index=0,
    help="Which error model to use"
)
lead_time = '0d' if 'Same-day' in lead_time_option else '1d'

st.sidebar.markdown("---")
refresh_btn = st.sidebar.button("🔄 Refresh Data")

# Load error model
error_model = load_error_model(lead_time)

if error_model is None:
    st.error("❌ Error model not found. Run: `python scripts/analysis/check_error_distribution.py`")
    st.stop()

# Fetch data
with st.spinner("Loading forecast and market data..."):
    forecast_df = fetch_openmeteo_forecast(selected_date)
    odds_df = fetch_polymarket_odds(selected_date)
    nws_forecast_df = fetch_nws_forecast()
    vc_forecast_df = fetch_visual_crossing_forecast(selected_date)
    actual_temps_df = fetch_actual_temperatures(selected_date)

# Check if data is available
if forecast_df is None:
    st.error(f"❌ No forecast data available for {selected_date}")
    days_ahead = (selected_date - date.today()).days
    if days_ahead < 0:
        st.info("💡 Cannot get forecast for past dates. Select today or a future date.")
    elif days_ahead == 0:
        st.info("💡 Open-Meteo forecast API doesn't include today's data (it starts from tomorrow).")
        st.info("📅 Please select tomorrow or a future date.")
        st.info(f"Try: {(date.today() + timedelta(days=1)).strftime('%B %d, %Y')}")
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

# Generate recommendations
recommendations = generate_recommendations(
    forecast_df, odds_df, error_model, selected_date, bankroll,
    min_edge=min_edge, min_ev=0.05, min_volume=100,
    max_bet_vs_volume=0.10, kelly_fraction=kelly_fraction, max_bet_pct=max_bet_pct
)

# Display error model info
st.markdown("### 📊 Model Information")
col1, col2, col3, col4 = st.columns(4)

# Get forecasted max for ONLY the betting day
betting_day_forecast = forecast_df[forecast_df['date'] == selected_date] if forecast_df is not None else None
forecasted_max_betting_day = betting_day_forecast['temp_f'].max() if betting_day_forecast is not None and not betting_day_forecast.empty else None

with col1:
    if forecasted_max_betting_day is not None:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 0.8rem; color: #718096; text-transform: uppercase; letter-spacing: 0.8px;">Forecast Max ({selected_date.strftime('%b %d')})</div>
            <div style="font-size: 2rem; font-weight: 700; color: #667EEA; margin: 0.5rem 0;">{forecasted_max_betting_day:.1f}°F</div>
            <div style="font-size: 0.875rem; color: #A0AEC0;">Open-Meteo</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 0.8rem; color: #718096; text-transform: uppercase; letter-spacing: 0.8px;">Forecast Max</div>
            <div style="font-size: 2rem; font-weight: 700; color: #CBD5E0; margin: 0.5rem 0;">—</div>
            <div style="font-size: 0.875rem; color: #A0AEC0;">Not available</div>
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

# Display forecast max cards
col1, col2, col3 = st.columns(3)

with col1:
    if openmeteo_max is not None:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 0.8rem; color: #718096; text-transform: uppercase; letter-spacing: 0.8px;">Open-Meteo Max</div>
            <div style="font-size: 2rem; font-weight: 700; color: #FC8181; margin: 0.5rem 0;">{openmeteo_max:.1f}°F</div>
            <div style="font-size: 0.875rem; color: #A0AEC0;">For {selected_date.strftime('%b %d')}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 0.8rem; color: #718096; text-transform: uppercase; letter-spacing: 0.8px;">Open-Meteo Max</div>
            <div style="font-size: 2rem; font-weight: 700; color: #CBD5E0; margin: 0.5rem 0;">—</div>
            <div style="font-size: 0.875rem; color: #A0AEC0;">Not available</div>
        </div>
        """, unsafe_allow_html=True)

with col2:
    if nws_max is not None:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 0.8rem; color: #718096; text-transform: uppercase; letter-spacing: 0.8px;">NWS Max</div>
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

with col3:
    if vc_max is not None:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 0.8rem; color: #718096; text-transform: uppercase; letter-spacing: 0.8px;">Visual Crossing Max</div>
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
    
    # Define 72-hour window from current time
    now = datetime.now(NY_TZ)
    window_start = now - timedelta(hours=24)  # 24 hours ago
    window_end = now + timedelta(hours=48)    # 48 hours ahead
    
    # Define target date boundaries at midnight ET
    target_date_start = datetime.combine(selected_date, datetime.min.time()).replace(tzinfo=NY_TZ)
    target_date_end = datetime.combine(selected_date + timedelta(days=1), datetime.min.time()).replace(tzinfo=NY_TZ)
    
    # Add actual temperature readings (if available)
    if actual_temps_df is not None:
        # Filter to 72-hour window
        actual_window = actual_temps_df[
            (actual_temps_df['timestamp'] >= window_start) & 
            (actual_temps_df['timestamp'] <= now)
        ]
        
        if not actual_window.empty:
            # Split by date relative to target
            actual_before = actual_window[actual_window['date'] < selected_date]
            actual_target = actual_window[actual_window['date'] == selected_date]
            
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
    
    fig_forecast.update_layout(
        title=f"72-Hour Temperature Window - {selected_date.strftime('%B %d, %Y')}",
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

if len(bets_to_place) == 0:
    st.markdown("""
    <div class="no-bet-card">
        <h4 style="margin: 0 0 0.5rem 0;">❌ No Bets Recommended</h4>
        <p style="margin: 0;">No markets meet the minimum edge criteria. Either the market is fairly priced or there's insufficient edge to justify a bet.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.success(f"✅ Found {len(bets_to_place)} betting opportunities!")
    
    for idx, bet in bets_to_place.iterrows():
        liquidity_note = " [LIQUIDITY CAPPED]" if bet['liquidity_constrained'] else ""
        
        st.markdown(f"""
        <div class="bet-card">
            <h4 style="margin: 0 0 1rem 0; font-size: 1.25rem;">🎲 BET: {bet['threshold']}</h4>
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

recommendations_sorted = recommendations.copy()
recommendations_sorted['sort_key'] = recommendations_sorted['threshold'].apply(sort_key)
recommendations_sorted = recommendations_sorted.sort_values('sort_key')

# Create comparison chart
fig = go.Figure()

fig.add_trace(go.Bar(
    x=recommendations_sorted['threshold'],
    y=recommendations_sorted['model_prob'],
    name='Model Probability',
    marker_color='#667EEA'
))

fig.add_trace(go.Bar(
    x=recommendations_sorted['threshold'],
    y=recommendations_sorted['market_prob'],
    name='Market Probability',
    marker_color='#CBD5E0'
))

fig.update_layout(
    title="Model vs Market Probabilities",
    xaxis_title="Temperature Range (Lowest to Highest)",
    yaxis_title="Probability",
    yaxis_tickformat='.0%',
    barmode='group',
    height=400,
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)

# Data table
st.markdown("### 📋 Detailed Analysis")
display_df = recommendations_sorted[['threshold', 'model_prob', 'market_prob', 'edge', 'ev', 'volume', 'should_bet', 'bet_size']].copy()
display_df.columns = ['Range', 'Model Prob', 'Market Prob', 'Edge', 'EV', 'Volume', 'Bet?', 'Bet Size']
display_df['Model Prob'] = display_df['Model Prob'].apply(lambda x: f"{x:.1%}")
display_df['Market Prob'] = display_df['Market Prob'].apply(lambda x: f"{x:.1%}")
display_df['Edge'] = display_df['Edge'].apply(lambda x: f"{x:+.1%}")
display_df['EV'] = display_df['EV'].apply(lambda x: f"{x:+.1%}")
display_df['Volume'] = display_df['Volume'].apply(lambda x: f"${x:,.0f}")
display_df['Bet?'] = display_df['Bet?'].apply(lambda x: "✅ YES" if x else "❌ No")
display_df['Bet Size'] = display_df['Bet Size'].apply(lambda x: f"${x:.2f}" if x > 0 else "—")

st.dataframe(display_df, use_container_width=True, height=400)

# Disclaimer
st.markdown("---")
st.markdown("""
<div style="background: #FFF3CD; padding: 1rem; border-radius: 8px; border: 1px solid #FFE69C; color: #856404;">
    <strong>⚠️ Disclaimer:</strong> This tool is for educational purposes only. Betting involves risk of loss. 
    Past performance does not guarantee future results. The model may be inaccurate. Always bet responsibly and within your means.
</div>
""", unsafe_allow_html=True)

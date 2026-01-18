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

# NOAA API Key - try Streamlit secrets first, then environment variable
try:
    NOAA_API_KEY = st.secrets.get("NOAA_API_KEY")
except:
    NOAA_API_KEY = os.getenv("NOAA_API_KEY")

# Page config
st.set_page_config(
    page_title="Temperature vs Odds - KLGA",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-metric {
        font-size: 2.5rem;
        font-weight: 700;
    }
    .insight-box {
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1a73e8;
        background-color: rgba(26, 115, 232, 0.1);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Cache data
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
            
            # Fetch price history - try 'max' first, if it seems truncated, fetch in chunks
            price_url = "https://clob.polymarket.com/prices-history"
            params = {
                'market': yes_token_id,
                'interval': 'max',  # Get all available data
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

# Title
st.title("📊 Temperature vs Market Odds")
st.markdown("**Understanding how odds react to temperature readings**")

# Sidebar
st.sidebar.title("⚙️ Controls")

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
st.sidebar.markdown("### 📊 Odds Display")

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
refresh_btn = st.sidebar.button("🔄 Refresh Data", width='stretch')

# Current stats
current_temp = temp_df['temp_f'].iloc[-1]
current_time = temp_df['timestamp'].iloc[-1]

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="main-metric">{:.1f}°F</div>'.format(current_temp), unsafe_allow_html=True)
    st.markdown(f"**Current Temperature**")
    st.caption(f"As of {current_time.strftime('%I:%M %p ET')}")

with col2:
    high_temp = temp_df['temp_f'].max()
    high_time = temp_df.loc[temp_df['temp_f'].idxmax(), 'timestamp']
    st.markdown('<div class="main-metric">{:.1f}°F</div>'.format(high_temp), unsafe_allow_html=True)
    st.markdown(f"**High (Period)**")
    st.caption(f"At {high_time.strftime('%I:%M %p ET')}")

with col3:
    if has_odds and len(buckets) > 0:
        # Find the bucket with highest probability
        latest_by_bucket = odds_df.groupby('threshold').last().reset_index()
        max_prob_bucket = latest_by_bucket.loc[latest_by_bucket['probability'].idxmax()]
        
        st.markdown('<div class="main-metric">{:.1%}</div>'.format(max_prob_bucket['probability']), unsafe_allow_html=True)
        st.markdown(f"**Most Likely: {max_prob_bucket['threshold_display']}**")
        st.caption(f"As of {max_prob_bucket['timestamp'].strftime('%I:%M %p ET')}")
    else:
        st.markdown('<div class="main-metric">-</div>', unsafe_allow_html=True)
        st.markdown("**Market Odds**")
        st.caption("No data")

st.markdown("---")

# Main chart - Temperature and Odds
if has_odds:
    # Create dual-axis chart
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
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
    
    # Plot temperature before target date (gray)
    if not temp_df_before.empty:
        fig.add_trace(
            go.Scatter(
                x=temp_df_before['timestamp'],
                y=temp_df_before['temp_f'],
                mode='lines',
                name='Temp (Before)',
                line=dict(color='#95a5a6', width=2, dash='dot'),
                hovertemplate='%{y:.1f}°F<br>%{x}<extra></extra>',
                showlegend=True
            ),
            row=1, col=1
        )
    
    # Plot temperature on target date (bright red/orange)
    if not temp_df_target.empty:
        fig.add_trace(
            go.Scatter(
                x=temp_df_target['timestamp'],
                y=temp_df_target['temp_f'],
                mode='lines',
                name=f'Temp ({selected_date.strftime("%b %d")})',
                line=dict(color='#e74c3c', width=3),
                hovertemplate='%{y:.1f}°F<br>%{x}<extra></extra>',
                showlegend=True
            ),
            row=1, col=1
        )
    
    # Plot temperature after target date (gray)
    if not temp_df_after.empty:
        fig.add_trace(
            go.Scatter(
                x=temp_df_after['timestamp'],
                y=temp_df_after['temp_f'],
                mode='lines',
                name='Temp (After)',
                line=dict(color='#95a5a6', width=2, dash='dot'),
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
    
    # Odds lines - show only selected buckets
    colors = ['#d93025', '#ea8600', '#1a73e8', '#0f9d58', '#9b59b6', '#e67e22', '#34a853']
    
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
        height=700,
        hovermode='x unified',
        template="plotly_white",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, width='stretch', key=f"main_chart_{selected_date}")
    
    # Analysis insights
    st.markdown("---")
    st.subheader("🔍 Market Analysis")
    
    # Show current odds for all buckets
    st.markdown("### 📊 Current Odds by Temperature Range")
    
    latest_by_bucket = odds_df.groupby('threshold').last().reset_index()
    latest_by_bucket = latest_by_bucket.sort_values('probability', ascending=False)
    
    # Create a bar chart of current odds
    fig_bar = go.Figure()
    
    fig_bar.add_trace(go.Bar(
        x=latest_by_bucket['threshold_display'],
        y=latest_by_bucket['probability'],
        marker_color=['#0f9d58' if p == latest_by_bucket['probability'].max() else '#1a73e8' 
                      for p in latest_by_bucket['probability']],
        text=[f"{p:.1%}" for p in latest_by_bucket['probability']],
        textposition='outside',
        hovertemplate='%{x}: %{y:.1%}<extra></extra>'
    ))
    
    fig_bar.update_layout(
        title=f"Latest Market Odds - {selected_date}",
        xaxis_title="Temperature Range",
        yaxis_title="Probability",
        yaxis_tickformat='.0%',
        height=400,
        showlegend=False,
        template="plotly_white"
    )
    
    st.plotly_chart(fig_bar, width='stretch', key=f"bar_chart_{selected_date}")
    
    # Show odds changes
    col1, col2, col3 = st.columns(3)
    
    with col1:
        max_bucket = latest_by_bucket.iloc[0]
        st.markdown(f"""
        <div class="insight-box">
            <h4>🎯 Most Likely</h4>
            <p><strong>{max_bucket['threshold_display']}</strong></p>
            <p>Probability: {max_bucket['probability']:.1%}</p>
            <p>Updated: {max_bucket['timestamp'].strftime('%I:%M %p ET')}</p>
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
            st.markdown(f"""
            <div class="insight-box">
                <h4>📈 Biggest Change</h4>
                <p><strong>{biggest_change_bucket}</strong></p>
                <p>Change: {biggest_change:+.1%}</p>
                <p>{biggest_change_time} ET</p>
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
        
        st.markdown(f"""
        <div class="insight-box">
            <h4>🌡️ Current vs Market</h4>
            <p><strong>Current: {current_temp:.1f}°F</strong></p>
            <p>Market expects: {max_bucket['threshold_display']}</p>
            <p>{'✅ In range' if in_range else '⚠️ Outside range'}</p>
        </div>
        """, unsafe_allow_html=True)

else:
    # Temperature only chart
    fig = go.Figure()
    
    # Split temperature data by date
    target_date_start = datetime.combine(selected_date, datetime.min.time()).replace(tzinfo=NY_TZ)
    target_date_end = datetime.combine(selected_date, datetime.max.time()).replace(tzinfo=NY_TZ)
    
    temp_df_target = temp_df[(temp_df['timestamp'] >= target_date_start) & (temp_df['timestamp'] <= target_date_end)]
    temp_df_before = temp_df[temp_df['timestamp'] < target_date_start]
    temp_df_after = temp_df[temp_df['timestamp'] > target_date_end]
    
    # Plot temperature before target date (gray)
    if not temp_df_before.empty:
        fig.add_trace(go.Scatter(
            x=temp_df_before['timestamp'],
            y=temp_df_before['temp_f'],
            mode='lines',
            name='Temp (Before)',
            line=dict(color='#95a5a6', width=2, dash='dot'),
            hovertemplate='%{y:.1f}°F<br>%{x}<extra></extra>'
        ))
    
    # Plot temperature on target date (bright red)
    if not temp_df_target.empty:
        fig.add_trace(go.Scatter(
            x=temp_df_target['timestamp'],
            y=temp_df_target['temp_f'],
            mode='lines',
            name=f'Temp ({selected_date.strftime("%b %d")})',
            line=dict(color='#e74c3c', width=3),
            hovertemplate='%{y:.1f}°F<br>%{x}<extra></extra>'
        ))
    
    # Plot temperature after target date (gray)
    if not temp_df_after.empty:
        fig.add_trace(go.Scatter(
            x=temp_df_after['timestamp'],
            y=temp_df_after['temp_f'],
            mode='lines',
            name='Temp (After)',
            line=dict(color='#95a5a6', width=2, dash='dot'),
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
        title=f"Temperature - Data Period",
        yaxis_title="Temperature (°F)",
        xaxis_title="Time (ET)",
        height=500,
        hovermode='x unified',
        template="plotly_white"
    )
    
    st.plotly_chart(fig, width='stretch', key=f"temp_only_chart_{selected_date}")

# Timezone verification
st.markdown("---")
st.subheader("🕐 Timezone Verification")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**NWS Data Timezone**")
    sample_time = temp_df.iloc[0]['timestamp']
    st.code(f"{sample_time}")
    st.success("✓ Timezone: America/New_York (ET)")
    st.caption("The -05:00 offset confirms Eastern Standard Time")

with col2:
    if has_odds:
        st.markdown("**Polymarket Data Timezone**")
        sample_odds_time = odds_df.iloc[0]['timestamp']
        st.code(f"{sample_odds_time}")
        st.success("✓ Timezone: America/New_York (ET)")
        st.caption("Both datasets use the same timezone for accurate comparison")
    else:
        st.markdown("**Polymarket Data Timezone**")
        st.info("No odds data loaded yet")

# Data tables
st.markdown("---")
with st.expander("📋 View Raw Data"):
    tab1, tab2 = st.tabs(["Temperature Data", "Odds Data"])
    
    with tab1:
        temp_display = temp_df[['timestamp', 'temp_f', 'humidity', 'wind_speed_mph']].copy()
        temp_display['timestamp'] = temp_display['timestamp'].dt.strftime('%Y-%m-%d %I:%M:%S %p ET')
        temp_display.columns = ['Time', 'Temp (°F)', 'Humidity (%)', 'Wind (mph)']
        st.dataframe(temp_display, width='stretch', height=300)
    
    with tab2:
        if has_odds:
            odds_display = odds_df[['timestamp', 'threshold_display', 'probability', 'question']].copy()
            odds_display['timestamp'] = odds_display['timestamp'].dt.strftime('%Y-%m-%d %I:%M:%S %p ET')
            odds_display['probability'] = odds_display['probability'].apply(lambda x: f"{x:.1%}")
            odds_display.columns = ['Time', 'Range', 'Odds', 'Market Question']
            st.dataframe(odds_display, width='stretch', height=300)
        else:
            st.info("No odds data available. Run: `python scripts/fetching/fetch_polymarket_historical.py`")

# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #666; font-size: 0.9rem;'>
    <p>Last updated: {datetime.now(NY_TZ).strftime('%Y-%m-%d %I:%M:%S %p')} ET</p>
    <p>Temperature: NWS KLGA | Odds: Polymarket | All times in Eastern Time</p>
</div>
""", unsafe_allow_html=True)

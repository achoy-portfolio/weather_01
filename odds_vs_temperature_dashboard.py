"""
Combined Dashboard: NWS Temperature vs Polymarket Odds
See how odds changed as temperature readings came in

Run: streamlit run odds_vs_temperature_dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, timezone
import sys
import os

sys.path.insert(0, '.')

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

from src.data.weather_scraper import WeatherScraper, WeatherDataError

NY_TZ = ZoneInfo("America/New_York")

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
@st.cache_data(ttl=300)
def fetch_nws_data(hours):
    """Fetch NWS temperature observations"""
    scraper = WeatherScraper(station_id="KLGA")
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(hours=hours)
    
    try:
        df = scraper.fetch_raw_observations(start_dt, end_dt)
        df.index = df.index.tz_convert(NY_TZ)
        df = df.reset_index()
        df = df.rename(columns={'index': 'timestamp'})
        return df
    except WeatherDataError as e:
        st.error(f"Error fetching NWS data: {e}")
        return None

@st.cache_data(ttl=300)
def load_polymarket_odds(date_str):
    """Load Polymarket historical odds for a specific date"""
    if date_str == "January 17":
        odds_file = 'data/raw/polymarket_odds_history_jan17.csv'
    elif date_str == "January 18":
        odds_file = 'data/raw/polymarket_odds_history.csv'
    else:
        return None
    
    if os.path.exists(odds_file):
        df = pd.read_csv(odds_file, parse_dates=['timestamp'])
        # Ensure timezone aware
        if df['timestamp'].dt.tz is None:
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize('UTC').dt.tz_convert(NY_TZ)
        else:
            df['timestamp'] = df['timestamp'].dt.tz_convert(NY_TZ)
        return df
    return None

# Title
st.title("📊 Temperature vs Market Odds")
st.markdown("**Understanding how odds react to temperature readings**")

# Sidebar
st.sidebar.title("⚙️ Controls")

# Date selector for odds data
available_dates = []
if os.path.exists('data/raw/polymarket_odds_history_jan17.csv'):
    available_dates.append("January 17")
if os.path.exists('data/raw/polymarket_odds_history.csv'):
    available_dates.append("January 18")

if available_dates:
    selected_date = st.sidebar.selectbox(
        "Market Date",
        options=available_dates,
        index=0
    )
else:
    selected_date = None
    st.sidebar.warning("No odds data found")

time_range = st.sidebar.selectbox(
    "Temperature Time Range",
    options=["6 hours", "12 hours", "24 hours", "48 hours"],
    index=2
)

hours_map = {
    "6 hours": 6,
    "12 hours": 12,
    "24 hours": 24,
    "48 hours": 48
}
hours = hours_map[time_range]

show_all_buckets = st.sidebar.checkbox("Show all temperature buckets", value=True)

st.sidebar.markdown("---")
refresh_btn = st.sidebar.button("🔄 Refresh Data", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### About
Compare real-time temperature readings with market odds to understand:
- When odds shifted
- How temperature trends affected betting
- Timezone verification (all times in ET)
""")

# Fetch data
with st.spinner("Loading data..."):
    temp_df = fetch_nws_data(hours)
    odds_df = load_polymarket_odds(selected_date) if selected_date else None

if temp_df is None or len(temp_df) == 0:
    st.error("No temperature data available")
    st.stop()

# Check if we have odds data
has_odds = odds_df is not None and len(odds_df) > 0

if not has_odds:
    st.warning("⚠️ No Polymarket odds data found. Run the historical odds fetcher first:")
    st.code("python scripts/fetching/fetch_polymarket_historical.py", language="bash")
    st.info("Showing temperature data only for now...")
    
# Get unique buckets from odds data
if has_odds:
    buckets = sorted(odds_df['threshold'].unique(), key=lambda x: (
        0 if '≤' in str(x) else (
            100 if '≥' in str(x) else int(str(x).split('-')[0])
        )
    ))
else:
    buckets = []

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
    st.markdown(f"**High ({time_range})**")
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
    
    # Temperature line
    fig.add_trace(
        go.Scatter(
            x=temp_df['timestamp'],
            y=temp_df['temp_f'],
            mode='lines',
            name='Temperature',
            line=dict(color='#e74c3c', width=2),
            hovertemplate='%{y:.1f}°F<br>%{x}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Add temperature range bands
    if show_all_buckets and len(buckets) > 0:
        for bucket in buckets:
            bucket_str = str(bucket)
            if '-' in bucket_str:
                low, high = map(int, bucket_str.split('-'))
                fig.add_hrect(
                    y0=low, y1=high,
                    fillcolor="gray", opacity=0.05,
                    line_width=0,
                    row=1, col=1
                )
    
    # Odds lines - show all buckets
    colors = ['#d93025', '#ea8600', '#1a73e8', '#0f9d58', '#9b59b6', '#e67e22', '#34a853']
    
    for i, bucket in enumerate(buckets):
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
    
    st.plotly_chart(fig, use_container_width=True)
    
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
    
    st.plotly_chart(fig_bar, use_container_width=True)
    
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
    
    fig.add_trace(go.Scatter(
        x=temp_df['timestamp'],
        y=temp_df['temp_f'],
        mode='lines',
        name='Temperature',
        line=dict(color='#e74c3c', width=2),
        hovertemplate='%{y:.1f}°F<br>%{x}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f"Temperature - {time_range}",
        yaxis_title="Temperature (°F)",
        xaxis_title="Time (ET)",
        height=500,
        hovermode='x unified',
        template="plotly_white"
    )
    
    st.plotly_chart(fig, use_container_width=True)

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
        st.dataframe(temp_display, use_container_width=True, height=300)
    
    with tab2:
        if has_odds:
            odds_display = odds_df[['timestamp', 'threshold_display', 'probability', 'question']].copy()
            odds_display['timestamp'] = odds_display['timestamp'].dt.strftime('%Y-%m-%d %I:%M:%S %p ET')
            odds_display['probability'] = odds_display['probability'].apply(lambda x: f"{x:.1%}")
            odds_display.columns = ['Time', 'Range', 'Odds', 'Market Question']
            st.dataframe(odds_display, use_container_width=True, height=300)
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

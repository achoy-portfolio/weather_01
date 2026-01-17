"""
NWS Temperature Explorer Dashboard - Google Finance style
Interactive exploration of LaGuardia temperature readings

Run: streamlit run nws_explorer_dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta, timezone
import sys

sys.path.insert(0, '.')

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

from src.data.weather_scraper import WeatherScraper, WeatherDataError

# New York timezone
NY_TZ = ZoneInfo("America/New_York")

# Page config
st.set_page_config(
    page_title="NWS Temperature Explorer - KLGA",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Google Finance-like styling
st.markdown("""
<style>
    .main-metric {
        font-size: 3.5rem;
        font-weight: 700;
        line-height: 1;
    }
    .metric-change {
        font-size: 1.2rem;
        font-weight: 500;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .stPlotlyChart {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Cache data fetching
@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_nws_data(hours):
    """Fetch NWS observations for KLGA"""
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
        st.error(f"Error fetching data: {e}")
        return None

# Title
st.title("🌡️ NWS Temperature Explorer")
st.markdown("**LaGuardia Airport (KLGA)** - Real-time 5-minute observations")

# Sidebar controls
st.sidebar.title("⚙️ Controls")

# Time range selector
time_range = st.sidebar.selectbox(
    "Time Range",
    options=["24 hours", "48 hours", "72 hours", "1 week", "2 weeks"],
    index=0
)

hours_map = {
    "24 hours": 24,
    "48 hours": 48,
    "72 hours": 72,
    "1 week": 168,
    "2 weeks": 336
}
hours = hours_map[time_range]

# Interval selector
interval_options = ["All data (5-min)", "15-min", "30-min", "Hourly"]
interval = st.sidebar.selectbox("Data Interval", options=interval_options, index=0)

# Chart type
chart_type = st.sidebar.radio(
    "Chart Style",
    options=["Line", "Candlestick", "Area"],
    index=0
)

# Additional metrics
show_humidity = st.sidebar.checkbox("Show Humidity", value=False)
show_wind = st.sidebar.checkbox("Show Wind Speed", value=False)
show_pressure = st.sidebar.checkbox("Show Pressure", value=False)

st.sidebar.markdown("---")
refresh_btn = st.sidebar.button("🔄 Refresh Data", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### About
Real-time temperature data from the National Weather Service.

**Station:** KLGA (LaGuardia Airport)  
**Update Frequency:** ~5 minutes  
**Source:** NWS API
""")

# Fetch data
with st.spinner("Loading data..."):
    df = fetch_nws_data(hours)

if df is None or len(df) == 0:
    st.error("No data available")
    st.stop()

# Resample data based on interval
if interval != "All data (5-min)":
    df_display = df.set_index('timestamp')
    
    if interval == "15-min":
        df_display = df_display.resample('15min').agg({
            'temp_f': ['first', 'max', 'min', 'last'],
            'humidity': 'mean',
            'wind_speed_mph': 'mean',
            'pressure_inhg': 'mean'
        })
    elif interval == "30-min":
        df_display = df_display.resample('30min').agg({
            'temp_f': ['first', 'max', 'min', 'last'],
            'humidity': 'mean',
            'wind_speed_mph': 'mean',
            'pressure_inhg': 'mean'
        })
    elif interval == "Hourly":
        df_display = df_display.resample('1h').agg({
            'temp_f': ['first', 'max', 'min', 'last'],
            'humidity': 'mean',
            'wind_speed_mph': 'mean',
            'pressure_inhg': 'mean'
        })
    
    df_display = df_display.dropna()
    df_display = df_display.reset_index()
else:
    df_display = df.copy()
    # For line chart, create OHLC-like structure
    df_display['temp_f_first'] = df_display['temp_f']
    df_display['temp_f_max'] = df_display['temp_f']
    df_display['temp_f_min'] = df_display['temp_f']
    df_display['temp_f_last'] = df_display['temp_f']

# Calculate statistics
current_temp = df['temp_f'].iloc[-1]
prev_temp = df['temp_f'].iloc[0]
temp_change = current_temp - prev_temp
temp_change_pct = (temp_change / prev_temp * 100) if prev_temp != 0 else 0

high_temp = df['temp_f'].max()
low_temp = df['temp_f'].min()
avg_temp = df['temp_f'].mean()

# Top metrics - Google Finance style
col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

with col1:
    change_color = "🟢" if temp_change >= 0 else "🔴"
    st.markdown(f'<div class="metric-label">Current Temperature</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="main-metric">{current_temp:.1f}°F</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="metric-change" style="color: {"#0f9d58" if temp_change >= 0 else "#d93025"}">'
        f'{change_color} {temp_change:+.1f}°F ({temp_change_pct:+.1f}%) {time_range}'
        f'</div>',
        unsafe_allow_html=True
    )

with col2:
    st.metric("High", f"{high_temp:.1f}°F", help=f"Highest in {time_range}")

with col3:
    st.metric("Low", f"{low_temp:.1f}°F", help=f"Lowest in {time_range}")

with col4:
    st.metric("Average", f"{avg_temp:.1f}°F", help=f"Average over {time_range}")

st.markdown("---")

# Main chart
if chart_type == "Candlestick" and interval != "All data (5-min)":
    # Candlestick chart (like stock chart)
    fig = go.Figure(data=[go.Candlestick(
        x=df_display['timestamp'],
        open=df_display[('temp_f', 'first')] if interval != "All data (5-min)" else df_display['temp_f_first'],
        high=df_display[('temp_f', 'max')] if interval != "All data (5-min)" else df_display['temp_f_max'],
        low=df_display[('temp_f', 'min')] if interval != "All data (5-min)" else df_display['temp_f_min'],
        close=df_display[('temp_f', 'last')] if interval != "All data (5-min)" else df_display['temp_f_last'],
        increasing_line_color='#0f9d58',
        decreasing_line_color='#d93025',
        name='Temperature'
    )])
    
    fig.update_layout(
        title=f"Temperature - {time_range}",
        yaxis_title="Temperature (°F)",
        xaxis_title="",
        height=500,
        hovermode='x unified',
        xaxis_rangeslider_visible=True,
        template="plotly_white"
    )

elif chart_type == "Area":
    # Area chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['temp_f'],
        fill='tozeroy',
        name='Temperature',
        line=dict(color='#1a73e8', width=2),
        fillcolor='rgba(26, 115, 232, 0.2)'
    ))
    
    fig.update_layout(
        title=f"Temperature - {time_range}",
        yaxis_title="Temperature (°F)",
        xaxis_title="",
        height=500,
        hovermode='x unified',
        template="plotly_white"
    )

else:
    # Line chart (default)
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['temp_f'],
        mode='lines',
        name='Temperature',
        line=dict(color='#1a73e8', width=2),
        hovertemplate='%{y:.1f}°F<br>%{x}<extra></extra>'
    ))
    
    # Add high/low markers
    max_idx = df['temp_f'].idxmax()
    min_idx = df['temp_f'].idxmin()
    
    fig.add_trace(go.Scatter(
        x=[df.loc[max_idx, 'timestamp']],
        y=[df.loc[max_idx, 'temp_f']],
        mode='markers',
        name='High',
        marker=dict(color='#d93025', size=10, symbol='triangle-up'),
        hovertemplate='High: %{y:.1f}°F<br>%{x}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=[df.loc[min_idx, 'timestamp']],
        y=[df.loc[min_idx, 'temp_f']],
        mode='markers',
        name='Low',
        marker=dict(color='#1967d2', size=10, symbol='triangle-down'),
        hovertemplate='Low: %{y:.1f}°F<br>%{x}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f"Temperature - {time_range}",
        yaxis_title="Temperature (°F)",
        xaxis_title="",
        height=500,
        hovermode='x unified',
        template="plotly_white",
        showlegend=True
    )

# Add range slider for zooming (Google Finance style)
fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1h", step="hour", stepmode="backward"),
            dict(count=6, label="6h", step="hour", stepmode="backward"),
            dict(count=12, label="12h", step="hour", stepmode="backward"),
            dict(count=1, label="1d", step="day", stepmode="backward"),
            dict(step="all", label="All")
        ]),
        bgcolor="rgba(150, 150, 150, 0.1)",
        activecolor="rgba(26, 115, 232, 0.2)"
    )
)

st.plotly_chart(fig, use_container_width=True)

# Additional metrics charts
if show_humidity or show_wind or show_pressure:
    st.markdown("---")
    st.subheader("📊 Additional Metrics")
    
    cols = st.columns(sum([show_humidity, show_wind, show_pressure]))
    col_idx = 0
    
    if show_humidity:
        with cols[col_idx]:
            fig_humidity = go.Figure()
            fig_humidity.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['humidity'],
                mode='lines',
                name='Humidity',
                line=dict(color='#0f9d58', width=2),
                fill='tozeroy',
                fillcolor='rgba(15, 157, 88, 0.1)'
            ))
            fig_humidity.update_layout(
                title="Humidity",
                yaxis_title="Humidity (%)",
                height=300,
                template="plotly_white",
                showlegend=False
            )
            st.plotly_chart(fig_humidity, use_container_width=True)
        col_idx += 1
    
    if show_wind:
        with cols[col_idx]:
            fig_wind = go.Figure()
            fig_wind.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['wind_speed_mph'],
                mode='lines',
                name='Wind Speed',
                line=dict(color='#9b59b6', width=2),
                fill='tozeroy',
                fillcolor='rgba(155, 89, 182, 0.1)'
            ))
            fig_wind.update_layout(
                title="Wind Speed",
                yaxis_title="Speed (mph)",
                height=300,
                template="plotly_white",
                showlegend=False
            )
            st.plotly_chart(fig_wind, use_container_width=True)
        col_idx += 1
    
    if show_pressure:
        with cols[col_idx]:
            fig_pressure = go.Figure()
            fig_pressure.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['pressure_inhg'],
                mode='lines',
                name='Pressure',
                line=dict(color='#e67e22', width=2),
                fill='tozeroy',
                fillcolor='rgba(230, 126, 34, 0.1)'
            ))
            fig_pressure.update_layout(
                title="Barometric Pressure",
                yaxis_title="Pressure (inHg)",
                height=300,
                template="plotly_white",
                showlegend=False
            )
            st.plotly_chart(fig_pressure, use_container_width=True)

# Data table
st.markdown("---")
with st.expander("📋 View Raw Data"):
    # Format for display
    df_table = df[['timestamp', 'temp_f', 'humidity', 'wind_speed_mph', 'pressure_inhg']].copy()
    df_table['timestamp'] = df_table['timestamp'].dt.strftime('%Y-%m-%d %I:%M:%S %p')
    df_table.columns = ['Time', 'Temp (°F)', 'Humidity (%)', 'Wind (mph)', 'Pressure (inHg)']
    
    st.dataframe(
        df_table.style.format({
            'Temp (°F)': '{:.1f}',
            'Humidity (%)': '{:.1f}',
            'Wind (mph)': '{:.1f}',
            'Pressure (inHg)': '{:.2f}'
        }),
        use_container_width=True,
        height=400
    )
    
    # Download button
    csv = df_table.to_csv(index=False)
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name=f"klga_temps_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

# Statistics summary
st.markdown("---")
st.subheader("📈 Statistics Summary")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Temperature**")
    st.write(f"Range: {low_temp:.1f}°F - {high_temp:.1f}°F")
    st.write(f"Spread: {high_temp - low_temp:.1f}°F")
    st.write(f"Std Dev: {df['temp_f'].std():.2f}°F")

with col2:
    st.markdown("**Humidity**")
    st.write(f"Average: {df['humidity'].mean():.1f}%")
    st.write(f"Range: {df['humidity'].min():.1f}% - {df['humidity'].max():.1f}%")

with col3:
    st.markdown("**Wind**")
    st.write(f"Average: {df['wind_speed_mph'].mean():.1f} mph")
    st.write(f"Max Gust: {df['wind_speed_mph'].max():.1f} mph")

# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #666; font-size: 0.9rem;'>
    <p>Last updated: {datetime.now(NY_TZ).strftime('%Y-%m-%d %I:%M:%S %p')} ET</p>
    <p>Data source: National Weather Service | Station: KLGA (LaGuardia Airport)</p>
</div>
""", unsafe_allow_html=True)

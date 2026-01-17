"""
Streamlit dashboard for Polymarket temperature betting.
Run: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os

# Add scripts to path
sys.path.insert(0, 'scripts')

# Import pipeline functions
from polymarket_pipeline import (
    load_model, 
    prepare_prediction_features,
    run_pipeline
)

# Page config
st.set_page_config(
    page_title="Polymarket Temperature Betting",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark mode compatibility
st.markdown("""
<style>
    .big-metric {
        font-size: 3rem !important;
        font-weight: bold;
        text-align: center;
    }
    .bet-card-strong {
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 3px solid #28a745;
        background-color: rgba(40, 167, 69, 0.1);
        margin: 1rem 0;
    }
    .bet-card-weak {
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 2px solid #ffc107;
        background-color: rgba(255, 193, 7, 0.1);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("⚙️ Settings")
bankroll = st.sidebar.number_input("Bankroll ($)", min_value=100, max_value=100000, value=1000, step=100)
min_edge = st.sidebar.slider("Min Edge (%)", min_value=1, max_value=20, value=5, step=1) / 100

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Forecast Weight")
forecast_weight = st.sidebar.slider(
    "NWS Forecast Weight (%)", 
    min_value=0, 
    max_value=100, 
    value=60, 
    step=5,
    help="How much to trust NWS forecast vs ML model. 100% = trust forecast completely, 0% = trust model only"
) / 100

st.sidebar.info(f"**Current blend:** {forecast_weight*100:.0f}% NWS + {(1-forecast_weight)*100:.0f}% ML Model")

auto_refresh = st.sidebar.checkbox("Auto-refresh (5 min)", value=False)

if auto_refresh:
    st.sidebar.info("Dashboard will refresh every 5 minutes")

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.markdown("""
This dashboard analyzes Polymarket temperature markets and identifies +EV betting opportunities.

**Data sources:**
- NWS 5-minute observations
- NWS forecast
- Polymarket odds
""")

# Main title
st.title("🌡️ Polymarket Temperature Betting Dashboard")
st.markdown(f"**Last updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Run pipeline
status_placeholder = st.empty()
status_placeholder.info("🔄 Running analysis...")

try:
    # Suppress stdout during pipeline run
    import io
    import contextlib
    
    # Capture stdout
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        opportunities = run_pipeline(
            bankroll=bankroll, 
            min_edge=min_edge,
            forecast_weight=forecast_weight
        )
    
    pipeline_output = f.getvalue()
    
    status_placeholder.success("✅ Analysis complete!")
    
    if opportunities and len(opportunities) > 0:
            # Extract prediction info from first opportunity
            pred = opportunities[0]['prediction']
            unc = opportunities[0]['uncertainty']
            
            # Top metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Predicted Temp", f"{pred:.1f}°F", f"±{unc:.1f}°F")
            
            with col2:
                conf_low = pred - 1.28 * unc
                conf_high = pred + 1.28 * unc
                st.metric("80% Confidence", f"{conf_low:.1f}-{conf_high:.1f}°F")
            
            with col3:
                bet_count = sum(1 for o in opportunities if o['recommendation'] == 'BET' or o['edge'] >= min_edge)
                st.metric("Opportunities", bet_count)
            
            with col4:
                total_kelly = sum(o['kelly_bet'] for o in opportunities if o['recommendation'] == 'BET' or o['edge'] >= min_edge)
                st.metric("Total Kelly", f"${total_kelly:.2f}", f"{total_kelly/bankroll*100:.1f}%")
            
            # Second row - Model breakdown
            st.markdown("---")
            st.markdown("### 🔍 Prediction Breakdown")
            
            col1, col2, col3 = st.columns(3)
            
            # Extract model components from first opportunity
            base_model = opportunities[0].get('base_model_pred', pred)
            nws_forecast = opportunities[0].get('forecast_high', pred)
            
            with col1:
                st.metric("🤖 ML Model Only", f"{base_model:.1f}°F", help="Based on historical patterns + today's temps")
            
            with col2:
                st.metric("🌤️ NWS Forecast", f"{nws_forecast:.1f}°F", help="National Weather Service forecast high")
            
            with col3:
                blend_text = f"{forecast_weight*100:.0f}% NWS + {(1-forecast_weight)*100:.0f}% ML"
                st.metric("📊 Final (Blended)", f"{pred:.1f}°F", help=blend_text)
            
            st.markdown("---")
            
            # Prediction distribution chart
            st.subheader("📊 Temperature Prediction Distribution")
            
            import numpy as np
            from scipy import stats
            
            x = np.linspace(pred - 4*unc, pred + 4*unc, 100)
            y = stats.norm.pdf(x, pred, unc)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x, y=y,
                fill='tozeroy',
                name='Probability',
                line=dict(color='#1f77b4', width=2)
            ))
            
            # Add confidence intervals
            fig.add_vline(x=pred, line_dash="dash", line_color="red", annotation_text="Prediction")
            fig.add_vrect(x0=conf_low, x1=conf_high, fillcolor="green", opacity=0.1, annotation_text="80% CI")
            
            fig.update_layout(
                xaxis_title="Temperature (°F)",
                yaxis_title="Probability Density",
                showlegend=False,
                height=300
            )
            
            st.plotly_chart(fig, width='stretch')
            
            st.markdown("---")
            
            # Betting opportunities
            st.subheader("🎯 Recommended Bets")
            
            bet_opps = [o for o in opportunities if o['recommendation'] == 'BET' or o['edge'] >= min_edge]
            
            if bet_opps:
                for i, opp in enumerate(bet_opps, 1):
                    threshold = opp['threshold']
                    edge_pct = opp['edge'] * 100
                    
                    # Get bet details
                    bet_side = opp.get('bet_side', 'N/A')
                    bet_price = opp.get('bet_price', 0)
                    market_question = opp.get('market_question', f'Temperature > {threshold}°F')
                    
                    # Extract range from question for display
                    import re
                    range_match = re.search(r'(\d+)-(\d+)°F', market_question)
                    if range_match:
                        range_display = f"{range_match.group(1)}-{range_match.group(2)}°F"
                    elif 'or below' in market_question.lower():
                        range_display = f"≤{threshold}°F"
                    elif 'or higher' in market_question.lower():
                        range_display = f"≥{threshold}°F"
                    else:
                        range_display = f">{threshold}°F"
                    
                    if edge_pct >= 10:
                        stars = "⭐⭐⭐"
                        badge = "🟢 Strong"
                    else:
                        stars = "⭐"
                        badge = "🟡 Moderate"
                    
                    # Create expandable card with range in title
                    title = f"{stars} Bet #{i}: {range_display} - {badge}"
                    with st.expander(title, expanded=(i <= 2)):
                        # Show bet instruction prominently at top
                        if bet_side == 'YES':
                            st.success(f"### ✅ BET YES at {bet_price:.1%}")
                        elif bet_side == 'NO':
                            st.error(f"### ❌ BET NO at {bet_price:.1%}")
                        else:
                            st.info(f"### Bet {bet_side} at {bet_price:.1%}")
                        
                        # Market question
                        st.markdown(f"**📋 Market:** {market_question}")
                        
                        st.markdown("---")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Market Odds", f"{opp['market_odds']:.1%}")
                            st.metric("Model Probability", f"{opp['model_prob']:.1%}")
                        
                        with col2:
                            st.metric("Edge", f"{edge_pct:+.1f}%")
                            st.metric("Expected Value", f"{opp['ev_pct']:+.1f}%")
                        
                        with col3:
                            st.metric("Kelly Bet", f"${opp['kelly_bet']:.2f}")
                            st.metric("% of Bankroll", f"{opp['kelly_pct']:.1f}%")
                        
                        st.info(f"💰 **Volume:** ${opp['volume']:,.0f}")
                
                # Summary
                total_bet = sum(o['kelly_bet'] for o in bet_opps)
                st.success(f"💰 **Total recommended:** ${total_bet:.2f} ({total_bet/bankroll*100:.1f}% of bankroll)")
            else:
                st.info("✅ No +EV betting opportunities found at current settings.")
                st.markdown("Try lowering the minimum edge threshold in the sidebar.")
            
            st.markdown("---")
            
            # Market comparison table
            st.subheader("📈 Market vs Model Comparison")
            
            comparison_data = []
            for opp in opportunities[:10]:  # Top 10
                comparison_data.append({
                    'Threshold': f">{opp['threshold']}°F",
                    'Market': f"{opp['market_odds']:.1%}",
                    'Model': f"{opp['model_prob']:.1%}",
                    'Edge': f"{opp['edge']:+.1%}",
                    'EV': f"{opp['ev_pct']:+.1f}%",
                    'Kelly': f"${opp['kelly_bet']:.2f}",
                    'Rec': opp['recommendation']
                })
            
            df_comparison = pd.DataFrame(comparison_data)
            
            # Color code recommendations
            def highlight_rec(row):
                if row['Rec'] == 'BET':
                    return ['background-color: rgba(40, 167, 69, 0.2)'] * len(row)
                else:
                    return [''] * len(row)
            
            st.dataframe(
                df_comparison.style.apply(highlight_rec, axis=1),
                width='stretch',
                hide_index=True
            )
        
    else:
        status_placeholder.warning("⚠️ No opportunities found")
        st.info("The pipeline returned no data. This could mean:")
        st.markdown("- Models are not trained")
        st.markdown("- Data sources are unavailable")
        st.markdown("- No markets available for tomorrow")
        
except Exception as e:
    status_placeholder.error("❌ Analysis failed")
    st.error(f"Error running pipeline: {str(e)}")
    with st.expander("Show error details"):
        st.exception(e)
        if 'pipeline_output' in locals():
            st.text("Pipeline output:")
            st.code(pipeline_output)

# Debug info at bottom
with st.expander("🔍 Debug Information"):
    if 'opportunities' in locals() and opportunities:
        st.write(f"**Total opportunities analyzed:** {len(opportunities)}")
        st.write(f"**Opportunities with +EV:** {sum(1 for o in opportunities if o['edge'] >= min_edge)}")
        st.json(opportunities[0] if len(opportunities) > 0 else {})
    
    if 'pipeline_output' in locals():
        st.text("Pipeline console output:")
        st.code(pipeline_output, language="text")

# Auto-refresh
if auto_refresh:
    import time
    time.sleep(300)  # 5 minutes
    st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9rem;'>
    <p>⚠️ For educational purposes only. Betting involves risk. Past performance doesn't guarantee future results.</p>
    <p>Data sources: NWS, Polymarket | Model: XGBoost with intraday features</p>
</div>
""", unsafe_allow_html=True)

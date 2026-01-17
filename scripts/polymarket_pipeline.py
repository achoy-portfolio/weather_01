"""
Complete pipeline for Polymarket temperature betting.
Runs: Model prediction -> Fetch NWS forecast -> Fetch Polymarket odds -> Analyze opportunities
"""

import sys
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats

# Import our modules
from fetch_nws_forecast import get_nws_forecast, get_daily_forecast_summary
from fetch_polymarket_odds import fetch_polymarket_event, get_todays_event_slug
from betting_strategy import estimate_probability_above_threshold, analyze_market_opportunity, calculate_kelly_bet_size

def load_model(model_file='models/peak_temp_intraday_xgboost.pkl'):
    """Load trained intraday model."""
    try:
        with open(model_file, 'rb') as f:
            data = pickle.load(f)
        print(f"✓ Model loaded: {model_file}")
        return data['model'], data['feature_cols']
    except FileNotFoundError:
        print(f"✗ Model not found: {model_file}")
        print("  Run: python scripts/train_intraday_model.py")
        return None, None

def load_probabilistic_model(model_file='models/probabilistic_temp_model.pkl'):
    """Load trained probabilistic model for uncertainty estimation."""
    try:
        with open(model_file, 'rb') as f:
            data = pickle.load(f)
        print(f"✓ Probabilistic model loaded: {model_file}")
        return data['models'], data['feature_cols'], data['quantiles']
    except FileNotFoundError:
        print(f"⚠ Probabilistic model not found: {model_file}")
        print("  Will use fixed uncertainty estimates")
        return None, None, None

def prepare_prediction_features(historical_file='data/raw/klga_hourly_full_history.csv', 
                                use_live_data=True):
    """Prepare features for today's prediction including intraday temperatures."""
    
    print("\nPreparing prediction features...")
    
    # Always load historical data first
    df = pd.read_csv(historical_file)
    df['valid'] = pd.to_datetime(df['valid'])
    df['date'] = df['valid'].dt.date
    df['hour'] = df['valid'].dt.hour
    
    # Get last 30 days from historical
    recent_df = df[df['valid'] >= (datetime.now() - timedelta(days=30))]
    
    # Aggregate to daily
    daily = recent_df.groupby('date').agg({
        'tmpf': ['mean', 'max']
    }).reset_index()
    daily.columns = ['date', 'temp_mean', 'temp_max']
    daily['date'] = pd.to_datetime(daily['date'])
    
    # Initialize today's intraday temps
    today = datetime.now().date()
    today_temps = {
        'temp_6am': None,
        'temp_9am': None,
        'temp_noon': None,
        'temp_3pm': None,
        'current_temp': None,
        'today_high_so_far': None
    }
    
    # Initialize tomorrow's forecast
    tomorrow_forecast = {
        'forecast_high': None,
        'forecast_low': None,
        'forecast_available': False
    }
    
    # Try to fetch live data to get today's intraday temps
    if use_live_data:
        print("  Fetching latest observations from NWS...")
        
        try:
            # Import the fetch function
            import sys
            sys.path.insert(0, '.')
            from scripts.nws_fetch_klga_5min import fetch_recent_observations
            
            # Fetch last 72 hours of observations
            live_df = fetch_recent_observations(hours=72)
            
            if live_df is not None and len(live_df) > 0:
                print(f"  ✓ Fetched {len(live_df)} recent observations")
                
                # Ensure timestamp column exists and convert
                if 'timestamp' in live_df.columns:
                    live_df['valid'] = pd.to_datetime(live_df['timestamp'])
                else:
                    live_df['valid'] = pd.to_datetime(live_df.index)
                
                live_df['date'] = live_df['valid'].dt.date
                live_df['hour'] = live_df['valid'].dt.hour
                
                # Extract today's temperatures at specific times
                today_data = live_df[live_df['date'] == today]
                
                if len(today_data) > 0:
                    # Get temps at specific hours (or closest available)
                    for target_hour, key in [(6, 'temp_6am'), (9, 'temp_9am'), 
                                              (12, 'temp_noon'), (15, 'temp_3pm')]:
                        hour_data = today_data[today_data['hour'] == target_hour]
                        if len(hour_data) > 0:
                            today_temps[key] = hour_data['temp_f'].iloc[0]
                    
                    # Current temp and today's high so far
                    today_temps['current_temp'] = today_data['temp_f'].iloc[-1]
                    today_temps['today_high_so_far'] = today_data['temp_f'].max()
                    
                    print(f"  ✓ Today's temps: 6am={today_temps['temp_6am']}, 9am={today_temps['temp_9am']}, "
                          f"noon={today_temps['temp_noon']}, 3pm={today_temps['temp_3pm']}")
                    print(f"  ✓ Current: {today_temps['current_temp']:.1f}°F, High so far: {today_temps['today_high_so_far']:.1f}°F")
                
                # Aggregate live data to daily for recent days
                daily_from_live = live_df.groupby('date').agg({
                    'temp_f': ['mean', 'max']
                }).reset_index()
                daily_from_live.columns = ['date', 'temp_mean', 'temp_max']
                daily_from_live['date'] = pd.to_datetime(daily_from_live['date'])
                
                # Merge with historical data (live data overwrites historical for same dates)
                daily = pd.concat([daily, daily_from_live]).drop_duplicates(subset=['date'], keep='last')
                daily = daily.sort_values('date')
                
                # Check yesterday's temp
                yesterday = datetime.now().date() - timedelta(days=1)
                yesterday_data_check = daily[daily['date'].dt.date == yesterday]
                if len(yesterday_data_check) > 0:
                    print(f"  ✓ Yesterday's actual temp: {yesterday_data_check['temp_max'].iloc[0]:.1f}°F")
                
            else:
                print(f"  ⚠ Could not fetch live data, using historical only")
        except Exception as e:
            print(f"  ⚠ Error fetching live data: {e}")
            print("  Using historical data only")
    
    # Fetch NWS forecast for tomorrow
    print("  Fetching NWS forecast for tomorrow...")
    try:
        from scripts.fetch_nws_forecast import get_nws_forecast, get_daily_forecast_summary
        
        forecast_df = get_nws_forecast()
        if forecast_df is not None and len(forecast_df) > 0:
            daily_forecast = get_daily_forecast_summary(forecast_df)
            
            if daily_forecast is not None and len(daily_forecast) > 0:
                # Get tomorrow's forecast
                tomorrow = datetime.now().date() + timedelta(days=1)
                tomorrow_fc = daily_forecast[daily_forecast['date'] == tomorrow]
                
                if len(tomorrow_fc) > 0:
                    tomorrow_forecast['forecast_high'] = tomorrow_fc['temp_max_forecast'].iloc[0]
                    tomorrow_forecast['forecast_low'] = tomorrow_fc['temp_min_forecast'].iloc[0]
                    tomorrow_forecast['forecast_available'] = True
                    print(f"  ✓ NWS forecast for tomorrow: High={tomorrow_forecast['forecast_high']:.1f}°F, "
                          f"Low={tomorrow_forecast['forecast_low']:.1f}°F")
                else:
                    print(f"  ⚠ No forecast found for tomorrow")
            else:
                print(f"  ⚠ Could not parse daily forecast")
        else:
            print(f"  ⚠ Could not fetch NWS forecast")
    except Exception as e:
        print(f"  ⚠ Error fetching forecast: {e}")
        print("  Continuing without forecast data")
    
    # Get yesterday's data
    yesterday = datetime.now().date() - timedelta(days=1)
    yesterday_data = daily[daily['date'].dt.date == yesterday]
    
    if len(yesterday_data) == 0:
        print(f"  ⚠ No data found for yesterday ({yesterday}), using most recent")
        yesterday_data = daily.tail(1)
        yesterday = yesterday_data['date'].iloc[0].date()
    
    # Calculate features for tomorrow's prediction
    tomorrow = datetime.now() + timedelta(days=1)
    
    features = {
        'month': tomorrow.month,
        'day_of_year': tomorrow.timetuple().tm_yday,
        'day_of_week': tomorrow.weekday(),
        'month_sin': np.sin(2 * np.pi * tomorrow.month / 12),
        'month_cos': np.cos(2 * np.pi * tomorrow.month / 12),
        'day_of_year_sin': np.sin(2 * np.pi * tomorrow.timetuple().tm_yday / 365),
        'day_of_year_cos': np.cos(2 * np.pi * tomorrow.timetuple().tm_yday / 365),
    }
    
    # Add lag features (yesterday, 2 days ago, etc.)
    for i, lag in enumerate([1, 2]):
        lag_date = yesterday - timedelta(days=lag-1)
        lag_data = daily[daily['date'].dt.date == lag_date]
        
        if len(lag_data) > 0:
            features[f'temp_mean_lag{lag}'] = lag_data['temp_mean'].iloc[0]
            features[f'temp_max_lag{lag}'] = lag_data['temp_max'].iloc[0]
        else:
            features[f'temp_mean_lag{lag}'] = yesterday_data['temp_mean'].iloc[0]
            features[f'temp_max_lag{lag}'] = yesterday_data['temp_max'].iloc[0]
    
    # Add min lag (use max as proxy if not available)
    features['temp_min_lag1'] = yesterday_data['temp_max'].iloc[0] - 10  # Rough estimate
    
    # Add dewpoint lag (use temp - 10 as rough estimate)
    features['dewpoint_mean_lag1'] = yesterday_data['temp_mean'].iloc[0] - 10
    features['dewpoint_mean_lag2'] = features['temp_mean_lag2'] - 10
    
    # Add humidity lag (rough estimate based on season)
    features['humidity_mean_lag1'] = 60  # Typical winter humidity
    features['humidity_mean_lag2'] = 60
    
    # Add rolling averages
    recent_temps = daily.tail(7)['temp_max'].values
    features['peak_temp_lag1'] = recent_temps[-1] if len(recent_temps) > 0 else 50
    features['peak_temp_lag2'] = recent_temps[-2] if len(recent_temps) > 1 else 50
    features['peak_temp_lag7'] = recent_temps[0] if len(recent_temps) > 6 else 50
    features['peak_temp_roll7'] = np.mean(recent_temps) if len(recent_temps) > 0 else 50
    features['peak_temp_roll7_std'] = np.std(recent_temps) if len(recent_temps) > 1 else 5
    
    # Add TODAY'S intraday features (key improvement!)
    # Use actual values if available, otherwise estimate from yesterday
    features['temp_6am'] = today_temps['temp_6am'] if today_temps['temp_6am'] else features['temp_min_lag1']
    features['temp_9am'] = today_temps['temp_9am'] if today_temps['temp_9am'] else features['temp_mean_lag1']
    features['temp_noon'] = today_temps['temp_noon'] if today_temps['temp_noon'] else features['temp_mean_lag1'] + 2
    features['temp_3pm'] = today_temps['temp_3pm'] if today_temps['temp_3pm'] else features['temp_max_lag1'] - 1
    
    # Calculate derived intraday features
    features['high_by_9am'] = max(features['temp_6am'], features['temp_9am'])
    features['high_by_noon'] = max(features['temp_6am'], features['temp_9am'], features['temp_noon'])
    features['high_by_3pm'] = max(features['temp_6am'], features['temp_9am'], features['temp_noon'], features['temp_3pm'])
    
    features['temp_change_6am_to_9am'] = features['temp_9am'] - features['temp_6am']
    features['temp_change_9am_to_noon'] = features['temp_noon'] - features['temp_9am']
    features['temp_change_noon_to_3pm'] = features['temp_3pm'] - features['temp_noon']
    
    # Add NWS forecast features (if available)
    if tomorrow_forecast['forecast_available']:
        features['forecast_high'] = tomorrow_forecast['forecast_high']
        features['forecast_low'] = tomorrow_forecast['forecast_low']
        features['forecast_range'] = tomorrow_forecast['forecast_high'] - tomorrow_forecast['forecast_low']
    else:
        # Use historical patterns as fallback
        features['forecast_high'] = features['peak_temp_lag1']  # Yesterday's high as proxy
        features['forecast_low'] = features['temp_min_lag1']
        features['forecast_range'] = features['forecast_high'] - features['forecast_low']
    
    print(f"  ✓ Features prepared for {tomorrow.strftime('%Y-%m-%d')}")
    print(f"  Yesterday ({yesterday}): {yesterday_data['temp_max'].iloc[0]:.1f}°F")
    print(f"  7-day avg: {features['peak_temp_roll7']:.1f}°F")
    if today_temps['current_temp']:
        print(f"  Today so far: Current={today_temps['current_temp']:.1f}°F, High={today_temps['today_high_so_far']:.1f}°F")
    if tomorrow_forecast['forecast_available']:
        print(f"  NWS forecast: {features['forecast_low']:.1f}-{features['forecast_high']:.1f}°F")
    
    return features

def run_pipeline(bankroll=1000, min_edge=0.05, forecast_weight=0.6):
    """
    Run complete betting pipeline.
    
    Args:
        bankroll: Total bankroll for Kelly sizing
        min_edge: Minimum edge required to recommend bet (default 5%)
        forecast_weight: Weight for NWS forecast (0-1, default 0.6 = 60%)
    """
    
    print("=" * 70)
    print("POLYMARKET TEMPERATURE BETTING PIPELINE")
    print("=" * 70)
    print(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Bankroll: ${bankroll:,.2f}")
    print(f"Min edge: {min_edge:.1%}")
    print("=" * 70)
    
    # Step 1: Load models
    print("\n[1/4] Loading models...")
    model, feature_cols = load_model()
    
    if model is None:
        print("\n✗ Pipeline failed: Model not found")
        return
    
    # Try to load probabilistic model for better uncertainty estimates
    prob_models, prob_feature_cols, quantiles = load_probabilistic_model()
    
    # Step 2: Prepare features
    print("\n[2/4] Preparing prediction features...")
    features = prepare_prediction_features()
    
    if features is None:
        print("\n✗ Pipeline failed: Could not prepare features")
        return
    
    # Convert features to DataFrame for prediction
    # Note: Model was trained without forecast features, so we'll use them
    # as additional context by creating a weighted prediction
    feature_df_base = pd.DataFrame([features])[feature_cols]
    
    # Get base model prediction
    base_pred = model.predict(feature_df_base)[0]
    
    # Store forecast for later
    forecast_high = features.get('forecast_high')
    has_forecast = forecast_high and forecast_high != features['peak_temp_lag1']
    
    # If we have a forecast, blend it with the model prediction
    if has_forecast:
        # Forecast is available and different from fallback
        forecast_high = features['forecast_high']
        
        # Use the provided forecast weight
        model_weight = 1 - forecast_weight
        
        final_pred = (forecast_weight * forecast_high) + (model_weight * base_pred)
        
        print(f"\n  Blending predictions:")
        print(f"    Model prediction: {base_pred:.1f}°F")
        print(f"    NWS forecast: {forecast_high:.1f}°F")
        print(f"    Final ({forecast_weight*100:.0f}% forecast, {model_weight*100:.0f}% model): {final_pred:.1f}°F")
    else:
        final_pred = base_pred
        print(f"\n  Using model prediction only: {final_pred:.1f}°F")
    
    # Estimate uncertainty based on probabilistic model or fallback to fixed values
    if prob_models is not None:
        # Use probabilistic model to estimate uncertainty
        print("\n  Estimating uncertainty using probabilistic model...")
        
        # Prepare features for probabilistic model (simpler feature set)
        prob_features = {
            'month': features['month'],
            'day_of_year': features['day_of_year'],
            'day_of_week': features['day_of_week'],
            'month_sin': features['month_sin'],
            'month_cos': features['month_cos'],
            'day_of_year_sin': features['day_of_year_sin'],
            'day_of_year_cos': features['day_of_year_cos'],
            'peak_temp_lag1': features['peak_temp_lag1'],
            'peak_temp_lag2': features['peak_temp_lag2'],
            'peak_temp_lag7': features['peak_temp_lag7'],
            'peak_temp_roll7': features['peak_temp_roll7'],
            'peak_temp_roll7_std': features['peak_temp_roll7_std'],
            'temp_mean_lag1': features['temp_mean_lag1'],
            'temp_mean_lag2': features['temp_mean_lag2'],
            'temp_max_lag1': features['temp_max_lag1'],
            'temp_max_lag2': features['temp_max_lag2'],
            'dewpoint_mean_lag1': features['dewpoint_mean_lag1'],
            'dewpoint_mean_lag2': features['dewpoint_mean_lag2'],
            'humidity_mean_lag1': features['humidity_mean_lag1'],
            'humidity_mean_lag2': features['humidity_mean_lag2']
        }
        
        prob_df = pd.DataFrame([prob_features])[prob_feature_cols]
        
        # Get quantile predictions
        q10_pred = prob_models[0.1].predict(prob_df)[0]
        q50_pred = prob_models[0.5].predict(prob_df)[0]
        q90_pred = prob_models[0.9].predict(prob_df)[0]
        
        # Estimate standard deviation from quantiles
        # Q90 - Q10 ≈ 2.56 * std for normal distribution
        estimated_std = (q90_pred - q10_pred) / 2.56
        uncertainty = estimated_std
        
        print(f"  Quantile predictions: Q10={q10_pred:.1f}°F, Q50={q50_pred:.1f}°F, Q90={q90_pred:.1f}°F")
        print(f"  Estimated uncertainty: ±{uncertainty:.1f}°F")
        
        # Adjust final prediction if using probabilistic model's median
        if not has_forecast:
            # If no forecast, use probabilistic model's median
            final_pred = q50_pred
            print(f"  Using probabilistic model median: {final_pred:.1f}°F")
    else:
        # Fallback to fixed uncertainty based on model performance
        if has_forecast:
            uncertainty = 4.0  # Roughly test MAE with forecast
        else:
            uncertainty = 6.0  # Roughly test RMSE without forecast
        forecast_high = None if not has_forecast else forecast_high
        print(f"  Using fixed uncertainty: ±{uncertainty:.1f}°F")
    
    # Step 3: Fetch Polymarket odds
    print("\n[3/4] Fetching Polymarket odds...")
    event_slug = get_todays_event_slug()
    print(f"  Event: {event_slug}")
    
    polymarket_data = fetch_polymarket_event(event_slug)
    
    if not polymarket_data:
        print("\n✗ Pipeline failed: Could not fetch Polymarket data")
        print("  Try manually specifying the event slug")
        return
    
    markets = polymarket_data['markets']
    print(f"✓ Found {len(markets)} markets")
    
    # Convert range probabilities to cumulative P(temp > threshold)
    # Polymarket has ranges like "34-35°F" with probability of landing in that range
    # We need to convert to "P(temp > X)"
    
    print("\n  Converting range probabilities to cumulative...")
    cumulative_markets = []
    
    # Sort markets by threshold
    sorted_markets = sorted(markets, key=lambda x: x['threshold'] if x['threshold'] else 0)
    
    for i, market in enumerate(sorted_markets):
        threshold = market['threshold']
        if threshold is None:
            continue
        
        # Check if this is a range market or a boundary market
        question = market['question'].lower()
        is_range = 'between' in question
        is_or_below = 'or below' in question or 'or lower' in question
        is_or_above = 'or higher' in question or 'or above' in question
        
        if is_or_below:
            # "33°F or below" with prob P means P(temp > 33) = 1 - P
            prob_above = 1 - market['yes_probability']
            cumulative_markets.append({
                'threshold': threshold,
                'prob_above': prob_above,
                'original_question': market['question'],
                'volume': market['volume']
            })
        elif is_or_above:
            # "44°F or higher" with prob P means P(temp > 44) = P (or close to it)
            # Actually P(temp >= 44), so P(temp > 43.5) approximately
            prob_above = market['yes_probability']
            cumulative_markets.append({
                'threshold': threshold,
                'prob_above': prob_above,
                'original_question': market['question'],
                'volume': market['volume']
            })
        elif is_range:
            # "34-35°F" - this is trickier, we need to sum all ranges above
            # P(temp > 35) = sum of all probabilities for ranges above 35
            prob_above = 0
            for j in range(i+1, len(sorted_markets)):
                if sorted_markets[j]['yes_probability']:
                    prob_above += sorted_markets[j]['yes_probability']
            
            cumulative_markets.append({
                'threshold': threshold,
                'prob_above': prob_above,
                'original_question': market['question'],
                'volume': market['volume']
            })
    
    print(f"  ✓ Converted to {len(cumulative_markets)} cumulative probabilities")
    
    # Display what we calculated
    print("\n  Market odds (cumulative):")
    for m in sorted(cumulative_markets, key=lambda x: x['threshold']):
        print(f"    P(temp > {m['threshold']}°F) = {m['prob_above']:.1%}")
    
    # Step 4: Analyze opportunities
    print("\n[4/4] Analyzing betting opportunities...")
    print("=" * 70)
    
    opportunities = []
    
    for market in cumulative_markets:
        threshold = market['threshold']
        market_odds = market['prob_above']
        
        # Predict using blended approach
        pred = final_pred
        
        # Calculate probability using normal distribution
        from scipy import stats
        z_score = (threshold - pred) / uncertainty
        model_prob = 1 - stats.norm.cdf(z_score)
        
        # Analyze opportunity
        analysis = analyze_market_opportunity(
            threshold, market_odds, model_prob, bankroll
        )
        
        if analysis:
            analysis['prediction'] = final_pred
            analysis['uncertainty'] = uncertainty
            analysis['volume'] = market['volume']
            analysis['market_question'] = market['original_question']
            analysis['base_model_pred'] = base_pred  # ML model only
            analysis['forecast_high'] = forecast_high  # NWS forecast
            opportunities.append(analysis)
    
    # Sort by expected value
    opportunities = sorted(opportunities, key=lambda x: x['ev_pct'], reverse=True)
    
    # Display results
    print("\nPREDICTION SUMMARY:")
    print("-" * 70)
    if opportunities:
        pred = opportunities[0]['prediction']
        unc = opportunities[0]['uncertainty']
        print(f"Predicted peak temperature: {pred:.1f}°F ± {unc:.1f}°F")
        print(f"80% confidence interval: [{pred - 1.28*unc:.1f}°F, {pred + 1.28*unc:.1f}°F]")
    print("-" * 70)
    
    # Create a mapping of thresholds to actual Polymarket markets
    print("\nPOLYMARKET MARKETS & RECOMMENDATIONS:")
    print("=" * 90)
    
    # Get original market data for display
    market_lookup = {}
    for market in markets:
        if market['threshold']:
            market_lookup[market['threshold']] = {
                'question': market['question'],
                'yes_price': market['yes_probability'],
                'volume': market['volume'],
                'direction': market.get('direction', 'unknown')
            }
    
    # Display each opportunity with clear betting instructions
    bet_recommendations = []
    
    for opp in opportunities:
        threshold = opp['threshold']
        if threshold not in market_lookup:
            continue
        
        market_info = market_lookup[threshold]
        question = market_info['question']
        
        # Determine what to bet
        # The cumulative prob is P(temp > threshold)
        # We need to map this back to the actual market
        
        is_range = 'between' in question.lower()
        is_or_below = 'or below' in question.lower()
        is_or_above = 'or higher' in question.lower()
        
        if opp['recommendation'] == 'BET' or opp['edge'] >= min_edge:
            # Figure out which side to bet
            model_prob_above = opp['model_prob']
            market_prob_above = opp['market_odds']
            
            if is_or_below:
                # Market: "33°F or below"
                # If model thinks higher chance of going above, bet NO on "or below"
                if model_prob_above > market_prob_above:
                    bet_side = "NO"
                    bet_price = 1 - market_info['yes_price']
                else:
                    bet_side = "YES"
                    bet_price = market_info['yes_price']
            elif is_or_above:
                # Market: "44°F or higher"
                # If model thinks higher chance of going above, bet YES
                if model_prob_above > market_prob_above:
                    bet_side = "YES"
                    bet_price = market_info['yes_price']
                else:
                    bet_side = "NO"
                    bet_price = 1 - market_info['yes_price']
            else:
                # Range market like "38-39°F"
                # This is trickier - the market is betting on landing IN that range
                # We need to calculate P(temp in range) from our model
                
                # Extract range from question
                import re
                range_match = re.search(r'(\d+)-(\d+)', question)
                if range_match:
                    range_low = int(range_match.group(1))
                    range_high = int(range_match.group(2))
                    
                    # Calculate model's P(temp in this range)
                    # P(low < temp <= high) = P(temp > low) - P(temp > high)
                    from scipy import stats
                    pred = opp['prediction']
                    unc = opp['uncertainty']
                    
                    z_low = (range_low - pred) / unc
                    z_high = (range_high - pred) / unc
                    
                    prob_above_low = 1 - stats.norm.cdf(z_low)
                    prob_above_high = 1 - stats.norm.cdf(z_high)
                    model_prob_in_range = prob_above_low - prob_above_high
                    
                    market_prob_in_range = market_info['yes_price']
                    
                    # If model thinks higher chance of landing in range, bet YES
                    if model_prob_in_range > market_prob_in_range:
                        bet_side = "YES"
                        bet_price = market_info['yes_price']
                    else:
                        bet_side = "NO"
                        bet_price = 1 - market_info['yes_price']
                else:
                    bet_side = "SKIP"
                    bet_price = market_info['yes_price']
            
            # Add to recommendations with full details
            bet_recommendations.append({
                'market': question,  # Full question, not truncated
                'bet_side': bet_side,
                'bet_price': bet_price,
                'bet_amount': opp['kelly_bet'],
                'edge': opp['edge'],
                'ev': opp['ev_pct'],
                'volume': market_info['volume'],
                'threshold': threshold
            })
            
            # Also add to the opportunity dict for dashboard
            opp['bet_side'] = bet_side
            opp['bet_price'] = bet_price
            opp['market_question'] = question
    
    if bet_recommendations:
        print("\n🎯 RECOMMENDED BETS:")
        print("=" * 100)
        
        for i, rec in enumerate(bet_recommendations, 1):
            market_str = rec['market']
            bet_str = rec['bet_side']
            price_str = f"{rec['bet_price']:.1%}"
            amount_str = f"${rec['bet_amount']:.2f}"
            edge_str = f"{rec['edge']:+.1%}"
            ev_str = f"{rec['ev']:+.1f}%"
            
            # Highlight strong bets
            if rec['edge'] >= 0.10:
                marker = "⭐⭐⭐"
            else:
                marker = "⭐"
            
            print(f"\n{marker} BET #{i}")
            print(f"  Market:  {market_str}")
            print(f"  Action:  Bet {bet_str} at {price_str}")
            print(f"  Amount:  {amount_str}")
            print(f"  Edge:    {edge_str} (EV: {ev_str})")
            print(f"  Volume:  ${rec['volume']:,.0f}")
        
        print("\n" + "=" * 100)
        total_bet = sum(r['bet_amount'] for r in bet_recommendations)
        print(f"Total recommended: ${total_bet:.2f} ({total_bet/bankroll*100:.1f}% of bankroll)")
    else:
        print("\n❌ No +EV betting opportunities found")
        print("Market odds align with model predictions")
    
    print("\n" + "=" * 90)
    
    # Show detailed technical analysis for reference
    print("\nDETAILED TECHNICAL ANALYSIS:")
    print("-" * 70)
    print(f"{'Threshold':<12} {'Market':<10} {'Model':<10} {'Edge':<10} {'EV':<10} {'Kelly':<12} {'Rec':<8}")
    print("-" * 70)
    print(f"{'Threshold':<12} {'Market':<10} {'Model':<10} {'Edge':<10} {'EV':<10} {'Kelly':<12} {'Rec':<8}")
    print("-" * 70)
    
    bet_count = 0
    total_kelly = 0
    
    for opp in opportunities:
        threshold_str = f">{opp['threshold']}°F"
        market_str = f"{opp['market_odds']:.1%}"
        model_str = f"{opp['model_prob']:.1%}"
        edge_str = f"{opp['edge']:+.1%}"
        ev_str = f"{opp['ev_pct']:+.1f}%"
        kelly_str = f"${opp['kelly_bet']:.2f}"
        rec = opp['recommendation']
        
        # Highlight good opportunities
        if opp['edge'] >= min_edge and opp['recommendation'] == 'BET':
            print(f"{threshold_str:<12} {market_str:<10} {model_str:<10} {edge_str:<10} {ev_str:<10} {kelly_str:<12} *** {rec}")
            bet_count += 1
            total_kelly += opp['kelly_bet']
        else:
            print(f"{threshold_str:<12} {market_str:<10} {model_str:<10} {edge_str:<10} {ev_str:<10} {kelly_str:<12} {rec}")
    
    print("-" * 70)
    print(f"\nRecommended bets: {bet_count}")
    print(f"Total Kelly allocation: ${total_kelly:.2f} ({total_kelly/bankroll*100:.1f}% of bankroll)")
    
    # Save results
    results_file = 'data/results/betting_opportunities.csv'
    
    # Create directory if it doesn't exist
    import os
    os.makedirs('data/results', exist_ok=True)
    
    df_results = pd.DataFrame(opportunities)
    df_results.to_csv(results_file, index=False)
    print(f"\n✓ Results saved to: {results_file}")
    
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    
    return opportunities

if __name__ == "__main__":
    # Run pipeline with default settings
    opportunities = run_pipeline(bankroll=1000, min_edge=0.05)

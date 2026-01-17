# Implementation Plan

- [x] 1. Set up project structure and dependencies

  - Create directory structure: `src/data`, `src/features`, `src/model`, `src/market`, `src/analysis`, `data/raw`, `data/processed`, `models`
  - Create `requirements.txt` with xgboost, pandas, numpy, scikit-learn, requests, beautifulsoup4, scipy, joblib
  - Create `__init__.py` files for all Python packages
  - _Requirements: 1.1, 2.1, 3.1, 4.1_

- [x] 2. Implement Weather Underground scraper

  - [x] 2.1 Create WeatherScraper class with station configuration

    - Implement `__init__` with station_id parameter (default KLGA)
    - Add request session with headers to mimic browser
    - Implement rate limiting (2 second delay between requests)
    - _Requirements: 1.1, 1.4_

  - [x] 2.2 Implement daily history fetch method

    - Parse Weather Underground history page HTML using BeautifulSoup
    - Extract max_temp, min_temp, humidity, wind_speed, precipitation from history table
    - Handle parsing errors gracefully with retry logic
    - _Requirements: 1.1, 1.2, 1.4_

  - [x] 2.3 Implement date range fetch and CSV export

    - Loop through date range with progress tracking
    - Aggregate results into pandas DataFrame
    - Save to CSV with date index
    - _Requirements: 1.3, 1.5_

  - [ ]\* 2.4 Write unit tests for weather scraper
    - Test HTML parsing with sample fixture
    - Test retry logic with mock failures
    - _Requirements: 1.1, 1.4_

- [ ] 3. Implement feature engineering pipeline

  - [ ] 3.1 Create FeatureEngineer class with temporal features
    - Implement `create_temporal_features` for day_of_year, month, week, is_weekend
    - Extract features from datetime index
    - _Requirements: 2.1_
  - [ ] 3.2 Implement lag and rolling features
    - Create `create_lag_features` for 1, 2, 3, 7 day lags
    - Create `create_rolling_features` for 3, 7, 14 day windows (mean and std)
    - _Requirements: 2.2, 2.3_
  - [ ] 3.3 Implement missing value handling and scaling
    - Implement `handle_missing_values` with forward-fill then backward-fill
    - Add StandardScaler for numerical feature normalization
    - Create `fit_transform` and `transform` methods
    - _Requirements: 2.4, 2.5_
  - [ ]\* 3.4 Write unit tests for feature engineering
    - Test temporal feature extraction
    - Test lag feature creation with known values
    - Test missing value imputation
    - _Requirements: 2.1, 2.2, 2.3, 2.5_

- [ ] 4. Implement XGBoost temperature model

  - [ ] 4.1 Create TemperatureModel class with training logic
    - Implement `__init__` with configurable hyperparameters
    - Implement `train` method with TimeSeriesSplit cross-validation (5 folds)
    - Calculate and return MAE and RMSE metrics
    - _Requirements: 3.1, 3.2, 3.3_
  - [ ] 4.2 Implement prediction and uncertainty estimation
    - Implement `predict` for point estimates
    - Implement `predict_with_uncertainty` using bootstrap sampling
    - Return prediction with 95% confidence interval bounds
    - _Requirements: 3.1, 5.2_
  - [ ] 4.3 Implement feature importance and model persistence
    - Implement `get_feature_importance` returning top 10 features
    - Implement `save` and `load` methods using joblib
    - _Requirements: 3.4, 3.5_
  - [ ]\* 4.4 Write unit tests for model
    - Test model training with sample data
    - Test prediction output shape
    - Test model save/load roundtrip
    - _Requirements: 3.1, 3.4_

- [ ] 5. Implement Polymarket client

  - [ ] 5.1 Create PolymarketClient class with caching
    - Implement `__init__` with cache dictionary and TTL (300 seconds)
    - Add request session for API calls
    - _Requirements: 4.4_
  - [ ] 5.2 Implement market data fetching
    - Implement `fetch_market_data` to retrieve market outcomes and prices
    - Parse Polymarket page or API response for temperature brackets
    - Handle network errors with cached fallback
    - _Requirements: 4.1, 4.2, 4.4_
  - [ ] 5.3 Implement probability parsing and storage
    - Implement `parse_temperature_brackets` to extract min/max temp ranges
    - Implement `get_implied_probabilities` returning DataFrame
    - Store odds with timestamps for historical tracking
    - _Requirements: 4.2, 4.3, 4.5_
  - [ ]\* 5.4 Write unit tests for Polymarket client
    - Test bracket parsing with mock market data
    - Test cache behavior
    - _Requirements: 4.1, 4.2_

- [ ] 6. Implement comparison analyzer and main script
  - [ ] 6.1 Create ComparisonAnalyzer class
    - Implement `calculate_model_probabilities` using normal distribution CDF
    - Map point prediction and std to probability per temperature bracket
    - _Requirements: 5.3_
  - [ ] 6.2 Implement probability comparison and opportunity detection
    - Implement `compare_probabilities` returning comparison DataFrame
    - Flag opportunities where model vs market difference exceeds 10 percentage points
    - Implement `generate_report` for human-readable output
    - _Requirements: 5.4, 5.5_
  - [ ] 6.3 Create main.py entry point
    - Wire together all components: scraper, feature engineering, model, polymarket, analyzer
    - Add CLI arguments for target date and market URL
    - Output prediction, confidence interval, and opportunity report
    - _Requirements: 5.1, 5.4, 5.5_
  - [ ]\* 6.4 Write integration tests
    - Test end-to-end pipeline with sample data
    - Test comparison output format
    - _Requirements: 5.1, 5.4, 5.5_

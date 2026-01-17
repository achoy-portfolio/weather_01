# Requirements Document

## Introduction

This feature builds an XGBoost machine learning model to predict the maximum daily temperature in New York City. The model will use historical weather data from Weather Underground and compare predictions against Polymarket odds for temperature-based prediction markets. The system will scrape historical weather data, train a predictive model, and provide temperature forecasts that can inform betting decisions on Polymarket.

## Glossary

- **XGBoost**: Extreme Gradient Boosting - a machine learning algorithm for regression and classification tasks
- **Weather Underground**: A weather data provider offering historical weather observations
- **Polymarket**: A prediction market platform where users can bet on real-world events
- **KLGA**: LaGuardia Airport weather station identifier used for NYC weather data
- **Max Temperature**: The highest recorded temperature during a 24-hour period
- **Feature Engineering**: The process of creating input variables for machine learning models
- **Prediction Market Odds**: Implied probabilities derived from market prices on Polymarket

## Requirements

### Requirement 1: Historical Weather Data Collection

**User Story:** As a data scientist, I want to collect historical weather data from Weather Underground, so that I can train a temperature prediction model.

#### Acceptance Criteria

1. WHEN the user initiates data collection, THE Weather_Data_Collector SHALL retrieve historical daily weather observations from Weather Underground for the KLGA station.
2. THE Weather_Data_Collector SHALL extract maximum temperature, minimum temperature, humidity, wind speed, and precipitation data for each historical day.
3. THE Weather_Data_Collector SHALL store collected data in a structured CSV format with date as the index column.
4. IF the Weather Underground API returns an error, THEN THE Weather_Data_Collector SHALL log the error and retry the request up to 3 times with exponential backoff.
5. THE Weather_Data_Collector SHALL collect a minimum of 365 days of historical data for model training.

### Requirement 2: Feature Engineering Pipeline

**User Story:** As a data scientist, I want to engineer relevant features from raw weather data, so that the model can learn meaningful patterns for temperature prediction.

#### Acceptance Criteria

1. THE Feature_Engineering_Pipeline SHALL create temporal features including day of year, month, and week number from the date column.
2. THE Feature_Engineering_Pipeline SHALL calculate rolling averages for temperature over 3-day, 7-day, and 14-day windows.
3. THE Feature_Engineering_Pipeline SHALL create lag features for temperature values from 1, 2, 3, and 7 days prior.
4. THE Feature_Engineering_Pipeline SHALL normalize numerical features to a standard scale before model training.
5. WHEN features contain missing values, THE Feature_Engineering_Pipeline SHALL impute missing values using forward-fill followed by backward-fill methods.

### Requirement 3: XGBoost Model Training

**User Story:** As a data scientist, I want to train an XGBoost regression model, so that I can predict maximum daily temperatures accurately.

#### Acceptance Criteria

1. THE XGBoost_Model SHALL be trained using historical weather features to predict maximum daily temperature.
2. THE XGBoost_Model SHALL use time-series cross-validation with a minimum of 5 folds to evaluate model performance.
3. THE XGBoost_Model SHALL report Mean Absolute Error (MAE) and Root Mean Square Error (RMSE) metrics after training.
4. THE XGBoost_Model SHALL save trained model weights to a file for future predictions.
5. WHEN training completes, THE XGBoost_Model SHALL generate a feature importance report showing the top 10 most influential features.

### Requirement 4: Polymarket Odds Integration

**User Story:** As a trader, I want to retrieve current Polymarket odds for NYC temperature events, so that I can compare model predictions against market expectations.

#### Acceptance Criteria

1. WHEN the user requests market data, THE Polymarket_Client SHALL retrieve current odds for NYC maximum temperature prediction markets.
2. THE Polymarket_Client SHALL parse temperature range brackets and their associated probabilities from market data.
3. THE Polymarket_Client SHALL convert market prices to implied probabilities for each temperature outcome.
4. IF the Polymarket API is unavailable, THEN THE Polymarket_Client SHALL return cached data from the most recent successful request.
5. THE Polymarket_Client SHALL store retrieved odds with timestamps for historical analysis.

### Requirement 5: Prediction and Comparison Output

**User Story:** As a trader, I want to see model predictions alongside Polymarket odds, so that I can identify potential betting opportunities.

#### Acceptance Criteria

1. THE Prediction_Engine SHALL generate a point estimate for the maximum temperature on a specified future date.
2. THE Prediction_Engine SHALL calculate a confidence interval for the temperature prediction using model uncertainty estimates.
3. THE Prediction_Engine SHALL map the predicted temperature to Polymarket temperature brackets and calculate implied probabilities.
4. THE Prediction_Engine SHALL display a comparison table showing model-implied probabilities versus Polymarket odds for each temperature bracket.
5. WHEN model probability differs from market probability by more than 10 percentage points, THE Prediction_Engine SHALL flag the bracket as a potential opportunity.

# Design Document: NYC Temperature Prediction for Polymarket

## Overview

This system predicts the maximum daily temperature in New York City using an XGBoost regression model trained on historical weather data from Weather Underground. The predictions are compared against Polymarket odds to identify potential betting opportunities where model-implied probabilities diverge significantly from market prices.

The system consists of four main components:

1. Weather data scraper for historical observations
2. Feature engineering pipeline for ML-ready data
3. XGBoost model training and prediction engine
4. Polymarket odds fetcher and comparison analyzer

## Architecture

```mermaid
flowchart TB
    subgraph Data Collection
        WU[Weather Underground API] --> Scraper[Weather Scraper]
        PM[Polymarket API] --> PMClient[Polymarket Client]
    end

    subgraph Data Processing
        Scraper --> RawData[(Raw Weather CSV)]
        RawData --> FE[Feature Engineering]
        FE --> Features[(Processed Features)]
    end

    subgraph Model
        Features --> Train[XGBoost Training]
        Train --> Model[(Saved Model)]
        Model --> Predict[Prediction Engine]
    end

    subgraph Output
        Predict --> Compare[Comparison Analyzer]
        PMClient --> Compare
        Compare --> Report[Opportunity Report]
    end
```

## Components and Interfaces

### 1. Weather Data Scraper (`src/data/weather_scraper.py`)

Responsible for fetching historical weather data from Weather Underground.

```python
class WeatherScraper:
    """Scrapes historical weather data from Weather Underground."""

    def __init__(self, station_id: str = "KLGA"):
        """Initialize scraper with weather station ID."""
        pass

    def fetch_daily_history(self, date: datetime.date) -> dict:
        """
        Fetch weather data for a specific date.

        Returns:
            dict with keys: date, max_temp, min_temp, avg_temp,
            humidity, wind_speed, precipitation, conditions
        """
        pass

    def fetch_date_range(self, start_date: date, end_date: date) -> pd.DataFrame:
        """
        Fetch weather data for a date range.

        Returns:
            DataFrame with daily weather observations
        """
        pass

    def save_to_csv(self, df: pd.DataFrame, filepath: str) -> None:
        """Save weather data to CSV file."""
        pass
```

**Implementation Notes:**

- Weather Underground URL pattern: `https://www.wunderground.com/history/daily/us/ny/new-york-city/KLGA/date/{YYYY-M-D}`
- Use BeautifulSoup or requests-html to parse the HTML response
- Extract data from the history table on the page
- Implement rate limiting (1 request per 2 seconds) to avoid blocking
- Cache responses locally to avoid redundant requests

### 2. Feature Engineering Pipeline (`src/features/feature_engineering.py`)

Transforms raw weather data into ML-ready features.

```python
class FeatureEngineer:
    """Creates features for temperature prediction model."""

    def __init__(self):
        self.scaler = StandardScaler()

    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add temporal features: day_of_year, month, week, is_weekend.
        """
        pass

    def create_lag_features(self, df: pd.DataFrame,
                           lags: list = [1, 2, 3, 7]) -> pd.DataFrame:
        """
        Create lagged temperature features.
        """
        pass

    def create_rolling_features(self, df: pd.DataFrame,
                                windows: list = [3, 7, 14]) -> pd.DataFrame:
        """
        Create rolling mean/std features for temperature.
        """
        pass

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values using forward-fill then backward-fill.
        """
        pass

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Full feature engineering pipeline with scaling.
        """
        pass

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted scaler.
        """
        pass
```

**Feature List:**
| Feature | Description | Type |
|---------|-------------|------|
| day_of_year | Day number (1-366) | Temporal |
| month | Month number (1-12) | Temporal |
| week | Week number (1-52) | Temporal |
| is_weekend | Saturday or Sunday flag | Temporal |
| temp_lag_1 | Max temp 1 day ago | Lag |
| temp_lag_2 | Max temp 2 days ago | Lag |
| temp_lag_3 | Max temp 3 days ago | Lag |
| temp_lag_7 | Max temp 7 days ago | Lag |
| temp_roll_3_mean | 3-day rolling mean | Rolling |
| temp_roll_7_mean | 7-day rolling mean | Rolling |
| temp_roll_14_mean | 14-day rolling mean | Rolling |
| temp_roll_7_std | 7-day rolling std | Rolling |
| humidity_avg | Average humidity | Weather |
| wind_speed_avg | Average wind speed | Weather |

### 3. XGBoost Model (`src/model/xgboost_model.py`)

Trains and manages the XGBoost regression model.

```python
class TemperatureModel:
    """XGBoost model for temperature prediction."""

    def __init__(self, params: dict = None):
        """
        Initialize model with hyperparameters.

        Default params:
            n_estimators: 100
            max_depth: 6
            learning_rate: 0.1
            subsample: 0.8
        """
        pass

    def train(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """
        Train model with time-series cross-validation.

        Returns:
            dict with 'mae', 'rmse', 'cv_scores'
        """
        pass

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate point predictions.
        """
        pass

    def predict_with_uncertainty(self, X: pd.DataFrame,
                                  n_iterations: int = 100) -> tuple:
        """
        Generate predictions with confidence intervals using
        bootstrap sampling.

        Returns:
            (point_estimate, lower_bound, upper_bound)
        """
        pass

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Return feature importance rankings.
        """
        pass

    def save(self, filepath: str) -> None:
        """Save model to file."""
        pass

    def load(self, filepath: str) -> None:
        """Load model from file."""
        pass
```

**Model Configuration:**

```python
DEFAULT_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'reg:squarederror',
    'random_state': 42
}
```

### 4. Polymarket Client (`src/market/polymarket_client.py`)

Fetches and parses Polymarket odds for temperature markets.

```python
class PolymarketClient:
    """Client for fetching Polymarket temperature market data."""

    def __init__(self):
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes

    def fetch_market_data(self, market_url: str) -> dict:
        """
        Fetch current market data from Polymarket.

        Returns:
            dict with 'outcomes', 'prices', 'volume', 'timestamp'
        """
        pass

    def parse_temperature_brackets(self, market_data: dict) -> list:
        """
        Parse temperature range brackets from market outcomes.

        Returns:
            list of dicts: [{'min': 30, 'max': 35, 'probability': 0.25}, ...]
        """
        pass

    def get_implied_probabilities(self, market_url: str) -> pd.DataFrame:
        """
        Get implied probabilities for each temperature bracket.

        Returns:
            DataFrame with columns: bracket, min_temp, max_temp, market_prob
        """
        pass
```

**Polymarket API Notes:**

- Polymarket uses a GraphQL API or REST endpoints
- Market URL pattern: `https://polymarket.com/event/{event-slug}`
- Prices are typically 0-1 representing probability
- Need to handle multiple outcome markets (temperature brackets)

### 5. Comparison Analyzer (`src/analysis/comparison.py`)

Compares model predictions with market odds.

```python
class ComparisonAnalyzer:
    """Analyzes model predictions vs market odds."""

    def __init__(self, model: TemperatureModel,
                 polymarket: PolymarketClient):
        self.model = model
        self.polymarket = polymarket

    def calculate_model_probabilities(self,
                                       prediction: float,
                                       std: float,
                                       brackets: list) -> dict:
        """
        Calculate probability of temperature falling in each bracket
        using normal distribution assumption.

        Returns:
            dict mapping bracket to probability
        """
        pass

    def compare_probabilities(self, market_url: str,
                              prediction_date: date) -> pd.DataFrame:
        """
        Generate comparison table of model vs market probabilities.

        Returns:
            DataFrame with columns: bracket, model_prob, market_prob,
            difference, opportunity_flag
        """
        pass

    def generate_report(self, comparison_df: pd.DataFrame) -> str:
        """
        Generate human-readable opportunity report.
        """
        pass
```

## Data Models

### Weather Observation

```python
@dataclass
class WeatherObservation:
    date: datetime.date
    max_temp: float  # Fahrenheit
    min_temp: float
    avg_temp: float
    humidity: float  # Percentage
    wind_speed: float  # mph
    precipitation: float  # inches
    conditions: str  # e.g., "Partly Cloudy"
```

### Market Outcome

```python
@dataclass
class MarketOutcome:
    bracket_name: str  # e.g., "35-39°F"
    min_temp: int
    max_temp: int
    price: float  # 0-1
    implied_probability: float
    volume: float
```

### Prediction Result

```python
@dataclass
class PredictionResult:
    target_date: datetime.date
    point_estimate: float
    lower_bound: float  # 95% CI
    upper_bound: float
    model_probabilities: dict  # bracket -> probability
    feature_contributions: dict  # feature -> contribution
```

## Error Handling

| Error Type                  | Handling Strategy                                   |
| --------------------------- | --------------------------------------------------- |
| Network timeout             | Retry 3 times with exponential backoff (2s, 4s, 8s) |
| Weather Underground blocked | Use cached data, log warning                        |
| Polymarket API error        | Return cached odds, flag as stale                   |
| Missing weather data        | Forward-fill, then backward-fill                    |
| Model prediction failure    | Return last known prediction with warning           |
| Invalid date range          | Raise ValueError with descriptive message           |

```python
class WeatherDataError(Exception):
    """Raised when weather data cannot be fetched."""
    pass

class MarketDataError(Exception):
    """Raised when market data cannot be fetched."""
    pass

class PredictionError(Exception):
    """Raised when prediction fails."""
    pass
```

## Testing Strategy

### Unit Tests

- Test feature engineering transformations with known inputs
- Test model prediction output shape and type
- Test probability calculations for temperature brackets
- Test CSV parsing and data validation

### Integration Tests

- Test end-to-end pipeline from data fetch to prediction
- Test model training with sample dataset
- Test comparison analyzer with mock market data

### Test Data

- Use historical weather data from January 2024 as test fixture
- Create mock Polymarket responses for testing
- Generate synthetic edge cases (extreme temperatures, missing data)

## File Structure

```
nyc-temp-prediction/
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── weather_scraper.py
│   ├── features/
│   │   ├── __init__.py
│   │   └── feature_engineering.py
│   ├── model/
│   │   ├── __init__.py
│   │   └── xgboost_model.py
│   ├── market/
│   │   ├── __init__.py
│   │   └── polymarket_client.py
│   └── analysis/
│       ├── __init__.py
│       └── comparison.py
├── data/
│   ├── raw/
│   │   └── weather_history.csv
│   └── processed/
│       └── features.csv
├── models/
│   └── xgboost_model.pkl
├── notebooks/
│   └── exploration.ipynb
├── tests/
│   ├── test_weather_scraper.py
│   ├── test_feature_engineering.py
│   ├── test_model.py
│   └── test_polymarket.py
├── main.py
├── requirements.txt
└── README.md
```

## Dependencies

```
xgboost>=1.7.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
requests>=2.31.0
beautifulsoup4>=4.12.0
scipy>=1.11.0
joblib>=1.3.0
```

# Requirements Document

## Introduction

This document specifies requirements for a backtesting system that evaluates the effectiveness of using Open-Meteo historical forecasts to inform Polymarket betting decisions on NYC temperature markets. The system will collect historical forecast data, actual temperature observations, and market odds to analyze betting accuracy and develop an error model for future betting strategies.

## Glossary

- **Backtest System**: The software system that evaluates historical betting performance using past forecast and market data
- **Open-Meteo Historical Forecast API**: API service that provides archived weather forecasts from past dates
- **Open-Meteo Archive API**: API service that provides actual observed weather data
- **Polymarket**: Prediction market platform where users bet on temperature outcomes
- **KLGA**: LaGuardia Airport weather station identifier (latitude: 40.7769, longitude: -73.8740)
- **Forecast Lead Time**: The number of days between when a forecast was issued and the target date
- **Market Open Time**: The time when Polymarket markets become available for betting (typically 2 days before the event)
- **Peak Temperature**: The maximum temperature recorded during a calendar day
- **Threshold Bucket**: A temperature range or boundary used in Polymarket markets (e.g., "≥75°F", "34-35°F")
- **Error Model**: A statistical model that quantifies the difference between forecasted and actual temperatures
- **Betting Strategy**: A decision framework that uses forecast data and error models to determine when and how much to bet

## Requirements

### Requirement 1

**User Story:** As a quantitative analyst, I want to collect historical Open-Meteo forecasts for each day from January 21, 2025 onward, so that I can analyze forecast accuracy over time.

#### Acceptance Criteria

1. WHEN THE Backtest System retrieves historical forecast data, THE Backtest System SHALL fetch forecasts from the Open-Meteo Historical Forecast API for KLGA coordinates
2. WHEN THE Backtest System processes a target date, THE Backtest System SHALL retrieve the forecast that was issued at 9 PM Eastern Time on the day before the target date
3. WHEN THE Backtest System collects forecast data, THE Backtest System SHALL store the forecasted peak temperature in degrees Fahrenheit for each target date
4. WHEN THE Backtest System encounters an API error, THE Backtest System SHALL log the error details and continue processing remaining dates
5. THE Backtest System SHALL retrieve historical forecasts for all dates from January 21, 2025 through the current date

### Requirement 2

**User Story:** As a quantitative analyst, I want to collect actual temperature observations for each day, so that I can compare forecasts against reality.

#### Acceptance Criteria

1. WHEN THE Backtest System retrieves actual temperature data, THE Backtest System SHALL fetch observations from the Open-Meteo Archive API for KLGA coordinates
2. WHEN THE Backtest System processes a target date, THE Backtest System SHALL extract the actual peak temperature in degrees Fahrenheit
3. WHEN THE Backtest System processes a target date, THE Backtest System SHALL extract the actual low temperature in degrees Fahrenheit
4. WHEN THE Backtest System processes a target date, THE Backtest System SHALL extract the actual average temperature in degrees Fahrenheit
5. THE Backtest System SHALL store actual temperature observations for all dates from January 21, 2025 through the current date

### Requirement 3

**User Story:** As a quantitative analyst, I want to collect historical Polymarket odds for each betting day, so that I can evaluate what betting opportunities were available.

#### Acceptance Criteria

1. WHEN THE Backtest System retrieves market odds, THE Backtest System SHALL fetch historical odds from Polymarket for NYC temperature markets
2. WHEN THE Backtest System processes a betting day, THE Backtest System SHALL retrieve odds for all threshold buckets available in the market
3. WHEN THE Backtest System collects odds data, THE Backtest System SHALL capture odds at the time when markets opened (approximately 2 days before the event)
4. WHEN THE Backtest System stores odds data, THE Backtest System SHALL record the probability for each threshold bucket in decimal format
5. THE Backtest System SHALL retrieve Polymarket odds for all dates from January 22, 2025 through January 10, 2026

### Requirement 4

**User Story:** As a quantitative analyst, I want to simulate betting decisions based on 9 PM forecasts, so that I can evaluate how accurate my betting strategy would have been.

#### Acceptance Criteria

1. WHEN THE Backtest System simulates a betting decision, THE Backtest System SHALL use the forecast issued at 9 PM Eastern Time on the day before the target date
2. WHEN THE Backtest System evaluates a threshold bucket, THE Backtest System SHALL determine whether the forecasted peak temperature falls within or exceeds the threshold
3. WHEN THE Backtest System compares forecast to actual outcome, THE Backtest System SHALL record whether the simulated bet would have won or lost
4. WHEN THE Backtest System calculates betting accuracy, THE Backtest System SHALL compute the percentage of correct predictions for each threshold type
5. THE Backtest System SHALL generate a simulated betting record for each day where both forecast and market data are available

### Requirement 5

**User Story:** As a quantitative analyst, I want to build an error model comparing predicted and actual peak temperatures, so that I can quantify forecast uncertainty.

#### Acceptance Criteria

1. WHEN THE Backtest System builds an error model, THE Backtest System SHALL calculate the difference between forecasted peak temperature and actual peak temperature for each day
2. WHEN THE Backtest System analyzes forecast errors, THE Backtest System SHALL compute the mean absolute error in degrees Fahrenheit
3. WHEN THE Backtest System analyzes forecast errors, THE Backtest System SHALL compute the root mean squared error in degrees Fahrenheit
4. WHEN THE Backtest System analyzes forecast errors, THE Backtest System SHALL compute the bias (mean error) in degrees Fahrenheit
5. WHEN THE Backtest System generates error statistics, THE Backtest System SHALL calculate error distributions by forecast lead time
6. THE Backtest System SHALL store error model parameters for use in future betting strategy optimization

### Requirement 6

**User Story:** As a quantitative analyst, I want to integrate error model uncertainty with market odds, so that I can develop a betting strategy that accounts for forecast reliability.

#### Acceptance Criteria

1. WHEN THE Backtest System evaluates a betting opportunity, THE Backtest System SHALL incorporate error model uncertainty into probability calculations
2. WHEN THE Backtest System compares model probability to market odds, THE Backtest System SHALL calculate the expected value of each potential bet
3. WHEN THE Backtest System identifies positive expected value, THE Backtest System SHALL recommend bet placement only when expected value exceeds 5 percent
4. WHEN THE Backtest System recommends bet sizing, THE Backtest System SHALL apply fractional Kelly criterion using error-adjusted probabilities
5. THE Backtest System SHALL generate a report showing how error-adjusted betting strategy would have performed historically

### Requirement 7

**User Story:** As a quantitative analyst, I want to store all collected data in structured formats, so that I can perform additional analysis and visualization.

#### Acceptance Criteria

1. WHEN THE Backtest System saves forecast data, THE Backtest System SHALL write data to CSV files in the data/raw directory
2. WHEN THE Backtest System saves actual temperature data, THE Backtest System SHALL write data to CSV files in the data/raw directory
3. WHEN THE Backtest System saves market odds data, THE Backtest System SHALL write data to CSV files in the data/raw directory
4. WHEN THE Backtest System saves backtest results, THE Backtest System SHALL write data to CSV files in the data/results directory
5. WHEN THE Backtest System writes CSV files, THE Backtest System SHALL include column headers with descriptive names
6. WHEN THE Backtest System writes CSV files, THE Backtest System SHALL use ISO 8601 date format for all date columns

### Requirement 8

**User Story:** As a quantitative analyst, I want to generate summary reports of backtest performance, so that I can quickly assess strategy effectiveness.

#### Acceptance Criteria

1. WHEN THE Backtest System completes analysis, THE Backtest System SHALL generate a summary report showing overall win rate
2. WHEN THE Backtest System generates a summary report, THE Backtest System SHALL display win rate broken down by threshold type
3. WHEN THE Backtest System generates a summary report, THE Backtest System SHALL display average forecast error by lead time
4. WHEN THE Backtest System generates a summary report, THE Backtest System SHALL display total simulated profit or loss
5. WHEN THE Backtest System generates a summary report, THE Backtest System SHALL display return on investment percentage

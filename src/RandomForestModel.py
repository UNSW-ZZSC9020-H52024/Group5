# -*- coding: utf-8 -*-
"""
Enhanced Energy Demand Prediction Model (Linear Regression)

Author: Manoj
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)

# 1. Load the datasets
temperature_data = pd.read_csv("../data/NSW/aggregated_temperature_data.csv")
humidity_data = pd.read_csv("../data/NSW/aggregated_humidity_data.csv")
wind_speed_data = pd.read_csv("../data/NSW/aggregated_windspeed_data.csv")
wind_direction_data = pd.read_csv("../data/NSW/aggregated_wind_direction_data.csv")
solar_radiation_data = pd.read_csv("../data/NSW/aggregated_solar_radiation_data.csv")
population_forecast_data = pd.read_csv("../data/NSW/PopulationForecastNSW.csv")
energy_demand_data = pd.read_csv("../data/NSW/totaldemand_nsw.csv")
enso_data = pd.read_csv("../data/NSW/daily_enso.csv")
soi_data = pd.read_csv("../data/NSW/soi_monthly.csv")

# 2. Preprocess the datasets
def preprocess_datetime(df, date_col='Date', time_col=None):
    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce')
    if time_col is not None:
        df[time_col] = df[time_col].astype(str)
        df['DateTime'] = pd.to_datetime(df[date_col].dt.strftime('%Y-%m-%d') + ' ' + df[time_col], errors='coerce')
    else:
        df['DateTime'] = df[date_col]
    df.dropna(subset=['DateTime'], inplace=True)
    return df

# Apply preprocessing to relevant datasets
temperature_data = preprocess_datetime(temperature_data, 'Date')
humidity_data = preprocess_datetime(humidity_data, 'Date')
wind_speed_data = preprocess_datetime(wind_speed_data, 'Date')
wind_direction_data = preprocess_datetime(wind_direction_data, 'Date')
solar_radiation_data = preprocess_datetime(solar_radiation_data, 'Date')

enso_data['DATE'] = pd.to_datetime(enso_data['DATE'], format='%m/%d/%Y', errors='coerce')
soi_data['yearmonth'] = soi_data['yearmonth'].astype(str) + '01'
soi_data['DateTime'] = pd.to_datetime(soi_data['yearmonth'], format='%Y%m%d', errors='coerce')
soi_data.dropna(subset=['DateTime'], inplace=True)

# Fix for non-numeric values in energy demand data
energy_demand_data['TOTALDEMAND'] = pd.to_numeric(energy_demand_data['TOTALDEMAND'], errors='coerce')
energy_demand_data.dropna(subset=['TOTALDEMAND'], inplace=True)

# Resample energy demand data from 30-minute intervals to daily
energy_demand_data['DATETIME'] = pd.to_datetime(energy_demand_data['DATETIME'], errors='coerce')
energy_demand_data.set_index('DATETIME', inplace=True)
energy_demand_data_daily = energy_demand_data.resample('D').mean()

# 3. Merge datasets on the common DateTime column
merged_data = temperature_data[['DateTime', 'mean_temp']].merge(
    humidity_data[['DateTime', 'mean_humidity']], on='DateTime', how='outer'
).merge(
    wind_speed_data[['DateTime', 'mean_windspeed']], on='DateTime', how='outer'
).merge(
    wind_direction_data[['DateTime', 'mean_wind_direction']], on='DateTime', how='outer'
).merge(
    solar_radiation_data[['DateTime', 'mean_solar_radiation']], on='DateTime', how='outer'
).merge(
    enso_data[['DATE', 'SOI']], left_on='DateTime', right_on='DATE', how='outer'
).merge(
    soi_data[['DateTime', 'soi']], on='DateTime', how='outer'
).merge(
    energy_demand_data_daily, left_on='DateTime', right_index=True, how='inner'
)

merged_data.drop(columns=['DATE'], inplace=True)

# Merge with population forecast data
population_forecast_data['Year'] = population_forecast_data['Year'].astype(int)
population_forecast_data = population_forecast_data[['Year', 'Medium_Series']]
merged_data['Year'] = merged_data['DateTime'].dt.year
merged_data = merged_data.merge(population_forecast_data, on='Year', how='left')

merged_data['Medium_Series'].fillna(method='ffill', inplace=True)

# 4. Handle missing values for other columns
merged_data.fillna(method='ffill', inplace=True)
merged_data.fillna(method='bfill', inplace=True)

# 5. Feature Engineering - create time-based features
merged_data['Hour'] = merged_data['DateTime'].dt.hour
merged_data['Day'] = merged_data['DateTime'].dt.day
merged_data['Month'] = merged_data['DateTime'].dt.month
merged_data['Week'] = merged_data['DateTime'].dt.isocalendar().week
merged_data['DayOfWeek'] = merged_data['DateTime'].dt.dayofweek

# 6. Create Lag Features
def create_lag_features(df, lags):
    for lag in lags:
        df[f'TOTALDEMAND_lag_{lag}'] = df['TOTALDEMAND'].shift(lag)
    return df

lags = [1, 24]
merged_data = create_lag_features(merged_data, lags)
merged_data.dropna(inplace=True)

# 7. Define features (X) and target (y)
features = [
    'mean_temp', 'mean_humidity', 'mean_windspeed', 'mean_wind_direction', 
    'mean_solar_radiation', 'SOI', 'Medium_Series', 'Hour', 'Day', 
    'Month', 'Week', 'DayOfWeek', 'TOTALDEMAND_lag_1', 'TOTALDEMAND_lag_24'
]
X = merged_data[features]
y = merged_data['TOTALDEMAND']

# Split the data
split_index = int(len(merged_data) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

# 8. Feature Scaling and Polynomial Features
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2, include_bias=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, features)
    ],
    remainder='passthrough'
)

# 9. Model Training and Evaluation (Linear Regression)
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('--- Linear Regression ---')
print(f'MSE: {mse}')
print(f'R-squared: {r2}')

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.plot(y_test.values[:100], label='Actual')
plt.plot(y_pred[:100], label='Predicted')
plt.title('Actual vs Predicted Energy Demand (Linear Regression)')
plt.legend()
plt.show()

# 10. Aggregation for Weekly, Monthly, and Yearly Predictions
def aggregate_predictions(y_test, y_pred, freq='W'):
    """Aggregate actual and predicted values based on the provided frequency."""
    y_test_resampled = y_test.resample(freq).sum()
    y_pred_resampled = pd.Series(y_pred, index=y_test.index).resample(freq).sum()
    return y_test_resampled, y_pred_resampled

# Function to plot aggregated predictions
def plot_aggregated_predictions(y_test, y_pred, freq_label):
    """Plot aggregated actual vs predicted values."""
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test.values, label=f'Actual ({freq_label})', color='blue')
    plt.plot(y_pred.index, y_pred.values, label=f'Predicted ({freq_label})', color='red')
    plt.xlabel('Date')
    plt.ylabel('Energy Demand')
    plt.title(f'Actual vs Predicted Energy Demand ({freq_label})')
    plt.legend()
    plt.show()

# Aggregating and plotting predictions for weekly, monthly, and yearly
frequencies = {
    'Weekly': 'W',
    'Monthly': 'M',
    'Yearly': 'Y'
}

for freq_label, freq in frequencies.items():
    y_test_resampled, y_pred_resampled = aggregate_predictions(y_test, y_pred, freq)
    plot_aggregated_predictions(y_test_resampled, y_pred_resampled, freq_label)

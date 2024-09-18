# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

# Load the CSV files
population_daily_path = '../data/Population/population_forecast_linear_daily.csv'
population_weekly_path = '../data/Population/population_forecast_linear_weekly.csv'
population_monthly_path = '../data/Population/population_forecast_linear_monthly.csv'
population_yearly_path = '../data/Population/population_forecast_linear_yearly.csv'
data_for_ml_path = '../data/NSW/data_for_ml.csv'

# Load each dataset
population_daily_df = pd.read_csv(population_daily_path)
population_weekly_df = pd.read_csv(population_weekly_path)
population_monthly_df = pd.read_csv(population_monthly_path)
population_yearly_df = pd.read_csv(population_yearly_path)
data_for_ml_df = pd.read_csv(data_for_ml_path)

# Convert the 'Date' columns to datetime for consistency
population_daily_df['Date'] = pd.to_datetime(population_daily_df['Date'], errors='coerce') 
population_weekly_df['Date'] = pd.to_datetime(population_weekly_df['Date'], errors='coerce')
population_monthly_df['Date'] = pd.to_datetime(population_monthly_df['Date'], errors='coerce')
population_yearly_df['Date'] = pd.to_datetime(population_yearly_df['Date'], errors='coerce')
data_for_ml_df['Date'] = pd.to_datetime(data_for_ml_df['Date'], errors='coerce')

# Drop the 'Population' and 'TOTALDEMAND' columns from data_for_ml_df
data_for_ml_df = data_for_ml_df.drop(columns=['Population', 'TOTALDEMAND'])

# Filter out incomplete years from the historical data
data_for_ml_df['Year'] = data_for_ml_df['Date'].dt.year
year_counts = data_for_ml_df['Year'].value_counts()
complete_years = year_counts[year_counts == 365].index
data_for_ml_df = data_for_ml_df[data_for_ml_df['Year'].isin(complete_years)]
data_for_ml_df = data_for_ml_df.drop(columns=['Year'])

# Get the forecast start and end dates from the daily population data
forecast_start_date = population_daily_df['Date'].min()
forecast_end_date = population_daily_df['Date'].max()

# Generate date ranges for daily, weekly, monthly, and yearly forecasts within the forecast period
forecast_dates_daily = pd.date_range(start=forecast_start_date, end=forecast_end_date, freq='D')
forecast_dates_weekly = pd.date_range(start=forecast_start_date, end=forecast_end_date, freq='W')
forecast_dates_monthly = pd.date_range(start=forecast_start_date, end=forecast_end_date, freq='ME')  # Month-end
forecast_dates_yearly = pd.date_range(start=forecast_start_date, end=forecast_end_date, freq='YE')   # Year-end

# Helper function to perform time series fill using complete yearly sets
def time_series_fill(feature_series, forecast_dates):
    """
    Repeats the yearly pattern of the feature based on available
    historical data for complete years and extends it to the forecast period.
    """
    # Convert to daily frequency and remove NaN values
    feature_series = feature_series.set_index('Date').asfreq('D').dropna()

    # Check the length of the feature series and the forecast dates
    feature_length = len(feature_series)
    forecast_length = len(forecast_dates)
    
    # Repeat the historical data enough times to cover the forecast period
    repeated_pattern = np.resize(feature_series.values, forecast_length)
    
    return repeated_pattern

# Prepare the forecast data for each frequency using time series fill
extrapolated_daily = pd.DataFrame(forecast_dates_daily, columns=['Date'])
extrapolated_weekly = pd.DataFrame(forecast_dates_weekly, columns=['Date'])
extrapolated_monthly = pd.DataFrame(forecast_dates_monthly, columns=['Date'])
extrapolated_yearly = pd.DataFrame(forecast_dates_yearly, columns=['Date'])

# List of features to extrapolate (exclude 'Date')
features_to_extrapolate = data_for_ml_df.columns.difference(['Date'])

# Apply time series fill to all features
for feature in features_to_extrapolate:
    feature_series = data_for_ml_df[['Date', feature]].copy()
    
    # Apply time series fill for each frequency
    extrapolated_daily[feature] = time_series_fill(feature_series, forecast_dates_daily)
    extrapolated_weekly[feature] = time_series_fill(feature_series, forecast_dates_weekly)
    extrapolated_monthly[feature] = time_series_fill(feature_series, forecast_dates_monthly)
    extrapolated_yearly[feature] = time_series_fill(feature_series, forecast_dates_yearly)

# Combine the population forecasts with extrapolated features (without Population and TOTALDEMAND columns)
population_daily_combined = pd.concat([population_daily_df.set_index('Date'), extrapolated_daily.set_index('Date')], axis=1).reset_index()
population_weekly_combined = pd.concat([population_weekly_df.set_index('Date'), extrapolated_weekly.set_index('Date')], axis=1).reset_index()
population_monthly_combined = pd.concat([population_monthly_df.set_index('Date'), extrapolated_monthly.set_index('Date')], axis=1).reset_index()
population_yearly_combined = pd.concat([population_yearly_df.set_index('Date'), extrapolated_yearly.set_index('Date')], axis=1).reset_index()

# Rename 'Population_Pred' to 'Population' after merging
population_daily_combined.rename(columns={'Population_Pred': 'Population'}, inplace=True)
population_weekly_combined.rename(columns={'Population_Pred': 'Population'}, inplace=True)
population_monthly_combined.rename(columns={'Population_Pred': 'Population'}, inplace=True)
population_yearly_combined.rename(columns={'Population_Pred': 'Population'}, inplace=True)

# Ensure 'Population' column is preserved and missing values are handled
population_daily_combined['Population'].fillna(method='ffill', inplace=True)
population_weekly_combined['Population'].fillna(method='ffill', inplace=True)
population_monthly_combined['Population'].fillna(method='ffill', inplace=True)
population_yearly_combined['Population'].fillna(method='ffill', inplace=True)

# Save the combined forecasts with time series filled features to CSV files
population_daily_combined.to_csv('../data/Population/population_forecast_with_features_daily.csv', index=False)
population_weekly_combined.to_csv('../data/Population/population_forecast_with_features_weekly.csv', index=False)
population_monthly_combined.to_csv('../data/Population/population_forecast_with_features_monthly.csv', index=False)
population_yearly_combined.to_csv('../data/Population/population_forecast_with_features_yearly.csv', index=False)

# Display a success message
print('Forecast data with time series fill saved successfully for daily, weekly, monthly, and yearly intervals.')

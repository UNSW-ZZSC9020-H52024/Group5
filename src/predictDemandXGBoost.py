
# -*- coding: utf-8 -*-
"""
Energy Demand Prediction Model (XGBoost with Aggregated Predictions)

Author: Manoj
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

# Step 1: Load Population Forecasts and rename 'Population_Pred' to 'Population'
population_forecast_daily = pd.read_csv('../data/Population/population_forecast_with_features_daily.csv')

# Rename 'Population_Pred' to 'Population'
population_forecast_daily.rename(columns={'Population_Pred': 'Population'}, inplace=True)
population_forecast_daily['Date'] = pd.to_datetime(population_forecast_daily['Date'])  # Convert 'Date' column to datetime

# Extract 'Year', 'Month', 'Week' from the 'Date' for aggregation purposes
population_forecast_daily['Year'] = population_forecast_daily['Date'].dt.year
population_forecast_daily['Month'] = population_forecast_daily['Date'].dt.month
population_forecast_daily['Week'] = population_forecast_daily['Date'].dt.isocalendar().week

# Features for the model
features = [
    'mean_temp', 'SOI', 'SST_DIFF', 'mean_humidity', 'mean_windspeed', 'mean_solar_radiation',
    'mean_wind_direction', 'rainfall', 'Population', 'DAYOFWEEK', 'DAYOFYEAR'
]

# Step 2: Prepare data for XGBoost model prediction
def prepare_forecast_data(population_forecast, features):
    X = population_forecast[features].values
    return X

# Prepare daily data
X_daily = prepare_forecast_data(population_forecast_daily, features)

# Step 3: Normalize the data
scaler = StandardScaler()
X_daily = scaler.fit_transform(X_daily)

# Step 4: Load the saved XGBoost model
model_path = '../data/NSW/saved_tuned_xgb_model.pkl'  # Define where the XGBoost model is saved
with open(model_path, 'rb') as f:
    best_xgb = pickle.load(f)

# Step 5: Make predictions for daily data
y_pred_daily = best_xgb.predict(X_daily)

# Assign predicted values to DataFrame
population_forecast_daily['Predicted_Demand'] = y_pred_daily

# Step 6: Aggregate the daily predictions to weekly, monthly, and yearly
population_forecast_weekly = population_forecast_daily.groupby(['Year', 'Week'], as_index=False).agg({'Predicted_Demand': 'sum'})
population_forecast_monthly = population_forecast_daily.groupby(['Year', 'Month'], as_index=False).agg({'Predicted_Demand': 'sum'})
population_forecast_yearly = population_forecast_daily.groupby(['Year'], as_index=False).agg({'Predicted_Demand': 'sum'})

# Create a date column for weekly data
population_forecast_weekly['Date'] = population_forecast_weekly.apply(lambda row: pd.to_datetime(f'{int(row["Year"])}-{int(row["Week"])}-1', format='%Y-%W-%w'), axis=1)

# Create a date column for monthly data
population_forecast_monthly['Date'] = population_forecast_monthly.apply(lambda row: pd.to_datetime(f'{int(row["Year"])}-{int(row["Month"])}-01'), axis=1)

# Use the first day of the year for yearly data
population_forecast_yearly['Date'] = population_forecast_yearly['Year'].apply(lambda year: pd.to_datetime(f'{int(year)}-01-01'))

# Step 7: Save the forecast results to CSV
population_forecast_daily.to_csv('../data/NSW/daily_demand_forecastXGB.csv', index=False)
population_forecast_weekly.to_csv('../data/NSW/weekly_demand_forecastXGB.csv', index=False)
population_forecast_monthly.to_csv('../data/NSW/monthly_demand_forecastXGB.csv', index=False)
population_forecast_yearly.to_csv('../data/NSW/yearly_demand_forecastXGB.csv', index=False)

print("XGBoost demand forecasts saved successfully to CSV files.")

# Step 8: Visualization - Subplots for Daily, Weekly, Monthly, and Yearly Forecasts

fig, ax = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

# Adjust line width and marker styles for better visualization
ax[0].plot(population_forecast_daily['Date'], population_forecast_daily['Predicted_Demand'], label='Daily Forecast', color='blue', linewidth=0.5, marker='o', markersize=3)
ax[0].set_title('Daily Predicted Energy Demand')
ax[0].set_ylabel('Predicted Demand')
ax[0].grid(True, which='both', linestyle='--', linewidth=0.5)
ax[0].legend()

ax[1].plot(population_forecast_weekly['Date'], population_forecast_weekly['Predicted_Demand'], label='Weekly Forecast', color='green', linewidth=0.8, marker='s', markersize=4)
ax[1].set_title('Weekly Predicted Energy Demand')
ax[1].set_ylabel('Predicted Demand')
ax[1].grid(True, which='both', linestyle='--', linewidth=0.5)
ax[1].legend()

ax[2].plot(population_forecast_monthly['Date'], population_forecast_monthly['Predicted_Demand'], label='Monthly Forecast', color='orange', linewidth=1, marker='^', markersize=5)
ax[2].set_title('Monthly Predicted Energy Demand')
ax[2].set_ylabel('Predicted Demand')
ax[2].grid(True, which='both', linestyle='--', linewidth=0.5)
ax[2].legend()

ax[3].plot(population_forecast_yearly['Date'], population_forecast_yearly['Predicted_Demand'], label='Yearly Forecast', color='red', linewidth=2, marker='D', markersize=6)
ax[3].set_title('Yearly Predicted Energy Demand')
ax[3].set_ylabel('Predicted Demand')
ax[3].set_xlabel('Date')
ax[3].grid(True, which='both', linestyle='--', linewidth=0.5)
ax[3].legend()

# Improve layout and show the plot
plt.tight_layout()
plt.show()

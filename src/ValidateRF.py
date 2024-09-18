# -*- coding: utf-8 -*-
"""
Energy Demand Prediction Model (Random Forest with Aggregated Predictions)

Author: Manoj
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import matplotlib.pyplot as plt

# Step 1: Load Data
def load_and_aggregate_data():
    daily_data = pd.read_csv('../data/NSW/data_for_ml.csv')
    daily_data['Date'] = pd.to_datetime(daily_data['Date'])
    daily_data['Year'] = daily_data['Date'].dt.year
    daily_data['Month'] = daily_data['Date'].dt.month
    daily_data['Week'] = daily_data['Date'].dt.isocalendar().week
    
    # Aggregate features
    daily_aggregated = daily_data.groupby('Date').mean(numeric_only=True).reset_index()
    weekly_aggregated = daily_data.groupby(['Year', 'Week']).mean(numeric_only=True).reset_index()
    monthly_aggregated = daily_data.groupby(['Year', 'Month']).mean(numeric_only=True).reset_index()
    yearly_aggregated = daily_data.groupby('Year').mean(numeric_only=True).reset_index()
    
    return daily_data, daily_aggregated, weekly_aggregated, monthly_aggregated, yearly_aggregated

# Step 2: Fill Missing Feature Data
def fill_missing_values(data, feature_columns):
    for col in feature_columns:
        if col in data.columns and data[col].isnull().sum() > 0:
            data[col] = data[col].fillna(data[col].mean())  # Fill with mean

# Step 3: Normalize Features
def normalize_features(data, features, scaler=None):
    X = data[features].values
    if scaler:
        X_scaled = scaler.transform(X)
    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

# Step 4: Main Execution
def main():
    # Load datasets
    daily_data, daily_aggregated, weekly_aggregated, monthly_aggregated, yearly_aggregated = load_and_aggregate_data()
    
    # Debugging: Print column names and missing value info
    print("Checking columns and missing values in datasets:")
    datasets = {'daily_data': daily_data, 'weekly_data': weekly_aggregated, 'monthly_data': monthly_aggregated, 'yearly_data': yearly_aggregated}
    for name, dataset in datasets.items():
        print(f"\n{name} columns: {dataset.columns}")
        print(f"{name} TOTALDEMAND missing values count: {dataset['TOTALDEMAND'].isnull().sum() if 'TOTALDEMAND' in dataset.columns else 'N/A'}")
        print(f"{name} TOTALDEMAND unique values: {dataset['TOTALDEMAND'].unique() if 'TOTALDEMAND' in dataset.columns else 'N/A'}")
        print(f"First few rows of {name}:\n{dataset.head()}")

    # Check for missing TOTALDEMAND values and handle them
    for name, dataset in datasets.items():
        if 'TOTALDEMAND' in dataset.columns:
            if dataset['TOTALDEMAND'].isnull().all():
                raise ValueError(f"TOTALDEMAND column is entirely NaN in {name}!")

    # Fill missing values for features
    feature_columns = ['mean_temp', 'SOI','SST_DIFF','mean_humidity', 'mean_windspeed','mean_solar_radiation',
    'mean_wind_direction', 'rainfall', 'Population','DAYOFWEEK','DAYOFYEAR']
    fill_missing_values(daily_data, feature_columns)
    fill_missing_values(weekly_aggregated, feature_columns)
    fill_missing_values(monthly_aggregated, feature_columns)
    fill_missing_values(yearly_aggregated, feature_columns)
    
    # Handle missing TOTALDEMAND values
    if 'TOTALDEMAND' in daily_data.columns:
        daily_data['TOTALDEMAND'] = daily_data['TOTALDEMAND'].ffill().bfill()
        missing_values_count = daily_data['TOTALDEMAND'].isnull().sum()
        total_rows = len(daily_data)
        print(f"Found {missing_values_count} missing values in TOTALDEMAND out of {total_rows} rows.")
        
        if missing_values_count > 0:
            print("Removing rows with missing TOTALDEMAND values.")
            daily_data = daily_data.dropna(subset=['TOTALDEMAND'])
        
        if daily_data.empty:
            raise ValueError("The dataset is empty after handling TOTALDEMAND values. Check your data preprocessing.")
    
    # Normalize features
    features = ['mean_temp','SOI','SST_DIFF', 'mean_humidity', 'mean_windspeed','mean_solar_radiation',
    'mean_wind_direction', 'rainfall', 'Population','DAYOFWEEK','DAYOFYEAR']
    
    X_daily, scaler = normalize_features(daily_data, features)
    X_yearly, scaler = normalize_features(yearly_aggregated, features, scaler)
    X_monthly, _ = normalize_features(monthly_aggregated, features, scaler)
    X_weekly, _ = normalize_features(weekly_aggregated, features, scaler)
    
    # Define and train the Random Forest model
    rf = RandomForestRegressor(random_state=42)
    param_distributions = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': [1.0, 'sqrt', 'log2']
    }
    
    random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_distributions, 
                                       n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)
    random_search.fit(X_daily, daily_data['TOTALDEMAND'])
    
    best_rf = random_search.best_estimator_
    
    # Save the trained model
    model_save_path = '../data/NSW/saved_tuned_rf_model.pkl'
    with open(model_save_path, 'wb') as f:
        pickle.dump(best_rf, f)
    print(f"Tuned Random Forest model saved to {model_save_path}")
    
    # Predict for daily, weekly, monthly, and yearly demand
    y_pred_daily = best_rf.predict(X_daily)
    y_pred_weekly = best_rf.predict(X_weekly)
    y_pred_monthly = best_rf.predict(X_monthly)
    y_pred_yearly = best_rf.predict(X_yearly)
    
    # Aggregate Daily Predictions
    daily_data['Predicted_Demand'] = y_pred_daily
    aggregated_daily_weekly = daily_data.groupby(['Year', 'Week']).sum(numeric_only=True)['Predicted_Demand'].reset_index()
    aggregated_daily_monthly = daily_data.groupby(['Year', 'Month']).sum(numeric_only=True)['Predicted_Demand'].reset_index()
    aggregated_daily_yearly = daily_data.groupby('Year').sum(numeric_only=True)['Predicted_Demand'].reset_index()
    
    # Combine Predictions
    def combine_predictions(data, aggregated_data, predictions, time_period):
        result = data[[time_period]].copy()
        result['Predicted_Demand'] = predictions
        result['Aggregated_Daily_Predicted_Demand'] = aggregated_data['Predicted_Demand']
        return result
    
    yearly_results = combine_predictions(yearly_aggregated, aggregated_daily_yearly, y_pred_yearly, 'Year')
    monthly_results = combine_predictions(monthly_aggregated, aggregated_daily_monthly, y_pred_monthly, 'Month')
    weekly_results = combine_predictions(weekly_aggregated, aggregated_daily_weekly, y_pred_weekly, 'Week')
    daily_results = daily_data[['Date', 'TOTALDEMAND', 'Predicted_Demand']].copy()

    # Plotting the results
    plt.figure(figsize=(12, 6))
    plt.plot(daily_data['Date'], daily_data['TOTALDEMAND'], label='Actual Demand', color='blue')
    plt.plot(daily_data['Date'], daily_data['Predicted_Demand'], label='Predicted Demand', color='red', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Demand')
    plt.title('Daily Energy Demand vs. Predicted Demand')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""
Enhanced Energy Demand Prediction Model (Linear Regression with Dask and Persisting Intermediate Results)

Author: Manoj
"""

import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

# Step 1: Setup Dask Client with Memory Limits
cluster = LocalCluster(memory_limit='16GB')  # Adjust memory limit as needed
client = Client(cluster)
print(f"Connected to Dask cluster: {client}")

# Step 2: Define helper functions for loading, filtering, and persisting data
def load_and_filter_data_dask(file_path, columns, date_col, start_date='2010-01-01', end_date='2021-03-18'):
    """Load only necessary columns, filter by date, and convert numeric columns to float32 using Dask."""
    df = dd.read_csv(file_path, usecols=columns, assume_missing=True)
    df[date_col] = dd.to_datetime(df[date_col], errors='coerce', dayfirst=True)
    df = df[(df[date_col] >= start_date) & (df[date_col] <= end_date)]
    
    # Persist intermediate data to disk
    output_file = f'./persisted_{os.path.basename(file_path)}.parquet'
    df = df.astype({col: np.float32 for col in df.select_dtypes(include=[np.number]).columns})
    df = df.persist()
    df.to_parquet(output_file, overwrite=True)
    
    return df

# Step 3: Load datasets with Dask and persist intermediate results
temperature_data = load_and_filter_data_dask("../data/NSW/aggregated_temperature_data.csv", ['Date', 'mean_temp'], 'Date')
humidity_data = load_and_filter_data_dask("../data/NSW/aggregated_humidity_data.csv", ['Date', 'mean_humidity'], 'Date')
wind_speed_data = load_and_filter_data_dask("../data/NSW/aggregated_windspeed_data.csv", ['Date', 'mean_windspeed'], 'Date')
wind_direction_data = load_and_filter_data_dask("../data/NSW/aggregated_wind_direction_data.csv", ['Date', 'mean_wind_direction'], 'Date')
solar_radiation_data = load_and_filter_data_dask("../data/NSW/aggregated_solar_radiation_data.csv", ['Date', 'mean_solar_radiation'], 'Date')
enso_data = load_and_filter_data_dask("../data/NSW/daily_enso.csv", ['DATE', 'SOI'], 'DATE')
soi_data = load_and_filter_data_dask("../data/NSW/soi_monthly.csv", ['yearmonth', 'soi'], 'yearmonth')

# Step 4: Load energy demand and population forecast data
energy_demand_data = load_and_filter_data_dask("../data/NSW/totaldemand_nsw.csv", ['DATETIME', 'TOTALDEMAND'], 'DATETIME')
population_forecast_data = dd.read_csv("../data/NSW/PopulationForecastNSW.csv", usecols=['Year', 'Medium_Series'])

# Step 5: Resample energy demand data and persist to disk
energy_demand_data['DATETIME'] = dd.to_datetime(energy_demand_data['DATETIME'], errors='coerce', dayfirst=True)
energy_demand_data = energy_demand_data.set_index('DATETIME').resample('D').mean()
energy_demand_data = energy_demand_data.persist()
energy_demand_data.to_parquet('./persisted_energy_demand_data.parquet', overwrite=True)

# Step 6: Define function for merging datasets in chunks and persisting results
def merge_datasets_in_chunks(left, right, on, how='outer'):
    """Helper function to merge datasets without using iloc."""
    return left.merge(right, on=on, how=how)

# Step 7: Merge datasets and persist intermediate results
enso_data = enso_data.rename(columns={'DATE': 'Date'})
soi_data = soi_data.rename(columns={'yearmonth': 'Date'})

merged_data = merge_datasets_in_chunks(temperature_data[['Date', 'mean_temp']],
                                       humidity_data[['Date', 'mean_humidity']], 'Date')

merged_data = merge_datasets_in_chunks(merged_data, wind_speed_data[['Date', 'mean_windspeed']], 'Date')
merged_data = merge_datasets_in_chunks(merged_data, wind_direction_data[['Date', 'mean_wind_direction']], 'Date')
merged_data = merge_datasets_in_chunks(merged_data, solar_radiation_data[['Date', 'mean_solar_radiation']], 'Date')
merged_data = merge_datasets_in_chunks(merged_data, enso_data[['Date', 'SOI']], 'Date', how='left')
merged_data = merge_datasets_in_chunks(merged_data, soi_data[['Date', 'soi']], 'Date', how='left')
energy_demand_data = energy_demand_data.reset_index().rename(columns={'DATETIME': 'Date'})  # Adjusting index for merging
merged_data = merge_datasets_in_chunks(merged_data, energy_demand_data[['Date', 'TOTALDEMAND']], 'Date', how='left')

# Persist merged data
merged_data = merged_data.persist()
merged_data.to_parquet('./persisted_merged_data.parquet', overwrite=True)

# Step 8: Merge with population forecast data
population_forecast_data['Year'] = population_forecast_data['Year'].astype(int)
merged_data['Year'] = dd.to_datetime(merged_data['Date']).dt.year
merged_data = merged_data.merge(population_forecast_data, on='Year', how='left')
merged_data['Medium_Series'] = merged_data['Medium_Series'].fillna(method='ffill')

# Handle missing values
merged_data = merged_data.fillna(method='ffill')
merged_data = merged_data.fillna(method='bfill')

# Persist final merged dataset
merged_data = merged_data.persist()
merged_data.to_parquet('./persisted_final_merged_data.parquet', overwrite=True)

# Step 9: Create Lag Features
def create_lag_features(df, column, lags):
    for lag in lags:
        df[f'{column}_lag_{lag}'] = df[column].shift(lag)
    return df

lags = [1, 24]
merged_data = create_lag_features(merged_data, 'TOTALDEMAND', lags)
merged_data = merged_data.dropna().persist()

# Step 10: Convert to float32
columns_to_convert = merged_data.columns.difference(['Date', 'Year', 'Medium_Series'])
merged_data[columns_to_convert] = merged_data[columns_to_convert].astype(np.float32)

# Step 11: Prepare for model training
features = [
    'mean_temp', 'mean_humidity', 'mean_windspeed', 'mean_wind_direction',
    'mean_solar_radiation', 'SOI', 'Medium_Series', 'TOTALDEMAND_lag_1', 'TOTALDEMAND_lag_24'
]
X = merged_data[features].compute().values
y = merged_data['TOTALDEMAND'].compute().values

# Step 12: Split the data
split_index = int(len(X) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Step 13: Move data to GPU using PyTorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

# Step 14: Define the PyTorch Linear Regression model
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)

# Step 15: Initialize model, optimizer, and loss function
model = LinearRegressionModel(X_train_tensor.shape[1]).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# Step 16: Train the model
def train_model(model, X_train, y_train, optimizer, loss_fn, epochs=100):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = loss_fn(y_pred, y_train)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f'Epoch {epoch}/{epochs}, Loss: {loss.item()}')

train_model(model, X_train_tensor, y_train_tensor, optimizer, loss_fn, epochs=100)

# Step 17: Evaluate the model
model.eval()
with torch.no_grad():
    y_pred_train = model(X_train_tensor)
    y_pred_test = model(X_test_tensor)

    train_mse = nn.MSELoss()(y_pred_train, y_train_tensor).item()
    test_mse = nn.MSELoss()(y_pred_test, y_test_tensor).item()

 
    print(f'Train MSE: {train_mse}')
    print(f'Test MSE: {test_mse}')

# 9. Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(y_test_tensor.cpu().numpy()[:100], label='Actual')
plt.plot(y_pred_test.cpu().numpy()[:100], label='Predicted')
plt.title('Actual vs Predicted Energy Demand (Linear Regression with PyTorch)')
plt.legend()
plt.show()

# Close the Dask client
client.close()
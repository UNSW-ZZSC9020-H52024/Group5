# -*- coding: utf-8 -*-
"""
Enhanced Energy Demand Prediction Model (Linear Regression with PyTorch using Dask)

Author: Manoj
"""

import dask.dataframe as dd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from dask.distributed import Client
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)

# Start Dask Client for better memory management
client = Client()

# 1. Load and Preprocess Data (Optimized with float32 to save memory using Dask)
def load_and_filter_data_dask(file_path, columns, date_col, start_date='2010-01-01', end_date='2021-03-18'):
    """Load necessary columns using Dask, filter by date, and convert numeric columns to float32."""
    df = dd.read_csv(file_path, usecols=columns, assume_missing=True)
    df[date_col] = dd.to_datetime(df[date_col], errors='coerce', dayfirst=True)
    df = df[(df[date_col] >= start_date) & (df[date_col] <= end_date)]
    
    # Convert numeric columns to float32 using map_partitions for Dask compatibility
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df = df.map_partitions(lambda partition: partition.astype({col: np.float32 for col in numeric_cols}))
    
    return df

# Load datasets using Dask with early filtering
temperature_data = load_and_filter_data_dask("../data/NSW/aggregated_temperature_data.csv", ['Date', 'mean_temp'], 'Date')
humidity_data = load_and_filter_data_dask("../data/NSW/aggregated_humidity_data.csv", ['Date', 'mean_humidity'], 'Date')
wind_speed_data = load_and_filter_data_dask("../data/NSW/aggregated_windspeed_data.csv", ['Date', 'mean_windspeed'], 'Date')
wind_direction_data = load_and_filter_data_dask("../data/NSW/aggregated_wind_direction_data.csv", ['Date', 'mean_wind_direction'], 'Date')
solar_radiation_data = load_and_filter_data_dask("../data/NSW/aggregated_solar_radiation_data.csv", ['Date', 'mean_solar_radiation'], 'Date')
enso_data = load_and_filter_data_dask("../data/NSW/daily_enso.csv", ['DATE', 'SOI'], 'DATE')
soi_data = load_and_filter_data_dask("../data/NSW/soi_monthly.csv", ['yearmonth', 'soi'], 'yearmonth')

# Load energy demand and population forecast data
energy_demand_data = load_and_filter_data_dask("../data/NSW/totaldemand_nsw.csv", ['DATETIME', 'TOTALDEMAND'], 'DATETIME')
population_forecast_data = pd.read_csv("../data/NSW/PopulationForecastNSW.csv", usecols=['Year', 'Medium_Series'])

# Resample energy demand data from 30-minute intervals to daily using Dask
energy_demand_data['DATETIME'] = dd.to_datetime(energy_demand_data['DATETIME'], errors='coerce')
energy_demand_data = energy_demand_data.set_index('DATETIME').resample('1D').mean()

# Persist datasets in memory to enable Dask to spill to disk if needed
temperature_data = temperature_data.persist()
humidity_data = humidity_data.persist()
wind_speed_data = wind_speed_data.persist()
wind_direction_data = wind_direction_data.persist()
solar_radiation_data = solar_radiation_data.persist()
enso_data = enso_data.persist()
soi_data = soi_data.persist()
energy_demand_data = energy_demand_data.persist()

# 2. Merging datasets using Dask
merged_data = dd.merge(temperature_data, humidity_data, on='Date', how='outer')
merged_data = dd.merge(merged_data, wind_speed_data, on='Date', how='outer')
merged_data = dd.merge(merged_data, wind_direction_data, on='Date', how='outer')
merged_data = dd.merge(merged_data, solar_radiation_data, on='Date', how='outer')
merged_data = dd.merge(merged_data, enso_data, left_on='Date', right_on='DATE', how='outer')
merged_data = dd.merge(merged_data, soi_data, left_on='Date', right_on='yearmonth', how='outer')
merged_data = dd.merge(merged_data, energy_demand_data, left_on='Date', right_index=True, how='inner')

# Drop unnecessary columns
merged_data = merged_data.drop(columns=['DATE', 'yearmonth'])

# Merge with population forecast data
merged_data['Year'] = dd.to_datetime(merged_data['Date']).dt.year
population_forecast_data['Year'] = population_forecast_data['Year'].astype(int)
population_forecast_dask = dd.from_pandas(population_forecast_data, npartitions=1)
merged_data = dd.merge(merged_data, population_forecast_dask, on='Year', how='left')

# Fill missing population data and forward-fill/bfill other missing values
merged_data['Medium_Series'] = merged_data['Medium_Series'].fillna(method='ffill')
merged_data = merged_data.fillna(method='ffill').fillna(method='bfill')

# 3. Create Lag Features
def create_lag_features(df, lags):
    for lag in lags:
        df[f'TOTALDEMAND_lag_{lag}'] = df['TOTALDEMAND'].shift(lag)
    return df

lags = [1, 24]
merged_data = create_lag_features(merged_data, lags)

# Persist merged data before computing to avoid recomputation
merged_data = merged_data.persist()

# Compute merged_data to a Pandas dataframe for further processing
merged_data = merged_data.compute()

# Apply dropna() on the Pandas dataframe
merged_data = merged_data.dropna()

# Define features (X) and target (y)
features = [
    'mean_temp', 'mean_humidity', 'mean_windspeed', 'mean_wind_direction', 
    'mean_solar_radiation', 'SOI', 'Medium_Series', 'TOTALDEMAND_lag_1', 'TOTALDEMAND_lag_24'
]
X = merged_data[features]
y = merged_data['TOTALDEMAND']

# 4. Split the data
split_index = int(len(merged_data) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

# Convert data to float32
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)
y_train = y_train.astype(np.float32)
y_test = y_test.astype(np.float32)

# 5. Move data to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1).to(device)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1).to(device)

# 6. Define the PyTorch Linear Regression model
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)

# Initialize the model, optimizer, and loss function
model = LinearRegressionModel(X_train_tensor.shape[1]).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# 7. Training the model (continued)
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

# Train the model
train_model(model, X_train_tensor, y_train_tensor, optimizer, loss_fn, epochs=100)

# 8. Evaluating the model
model.eval()
with torch.no_grad():
    y_pred_train = model(X_train_tensor)
    y_pred_test = model(X_test_tensor)

    train_mse = mean_squared_error(y_train_tensor.cpu(), y_pred_train.cpu())
    test_mse = mean_squared_error(y_test_tensor.cpu(), y_pred_test.cpu())

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

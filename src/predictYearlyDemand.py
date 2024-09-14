# -*- coding: utf-8 -*-
"""
Energy Demand Prediction Model (Linear Regression with PyTorch)

Author: Manoj
"""

import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import os

# Step 1: Load the population forecast dataset and other feature datasets  
yearly_population_forecast_path = '../data/Population/population_forecast_linear_yearly.csv'  # Replace with actual path
population_forecast_data = pd.read_csv(yearly_population_forecast_path)

# Debugging population forecast data
print("Population Forecast Data:")
print(population_forecast_data.head())

# If the population forecast data has a 'Date' column, convert it to 'Year'
if 'Date' in population_forecast_data.columns:
    population_forecast_data['Date'] = pd.to_datetime(population_forecast_data['Date'])
    population_forecast_data['Year'] = population_forecast_data['Date'].dt.year
    population_forecast_data.drop(columns=['Date'], inplace=True)  # Drop 'Date' if not needed

# Filter the population forecast data to only include future years (2023 onward)
future_population_data = population_forecast_data[population_forecast_data['Year'] >= 2023]

# Debugging filtered forecast data
print(f"Years in Population Forecast: {future_population_data['Year'].unique()}")

# Step 2: Aggregate your features (temperature, humidity, etc.) to yearly averages
daily_features_path = '../data/NSW/data_for_ml.csv'  # Replace with actual path
daily_data = pd.read_csv(daily_features_path)

# Convert 'Date' to datetime to extract 'Year'
daily_data['Date'] = pd.to_datetime(daily_data['Date'])
daily_data['Year'] = daily_data['Date'].dt.year

# Debugging daily data
print("Daily Data (Yearly Aggregation Preview):")
print(daily_data.head())

# Aggregate features by year (taking the mean)
yearly_data = daily_data.groupby('Year').mean(numeric_only=True).reset_index()

# Debugging yearly data before merge
print("Yearly Data Before Merge:")
print(yearly_data.head())

# Step 3: Ensure matching between forecast and feature data
# First, ensure that both datasets contain matching years
matching_years = future_population_data['Year'].isin(yearly_data['Year'])
if not matching_years.all():
    print("Some years from the forecast do not exist in the feature data. Adding missing years with NaNs.")

# Ensure both datasets contain all future years
all_years = pd.DataFrame({'Year': future_population_data['Year']})
yearly_data = all_years.merge(yearly_data, on='Year', how='left')

# Step 4: Merge the yearly population forecast with the yearly aggregated features
yearly_data = future_population_data.merge(yearly_data, on='Year', how='left')

# Step 5: Fill missing feature data for future years using historical averages
# Replace NaNs with historical means for each feature
feature_columns = ['mean_humidity', 'enso', 'mean_solar_radiation', 'mean_temp',
                   'mean_wind_direction', 'mean_windspeed', 'rainfall']

for col in feature_columns:
    if yearly_data[col].isnull().sum() > 0:
        yearly_data[col].fillna(daily_data[col].mean(), inplace=True)

# Debugging yearly data after filling missing values
print("Yearly Data Columns After Merge:", yearly_data.columns)
print("Yearly Data After Merge Preview:")
print(yearly_data.head())

# Ensure Population column is properly filled
yearly_data['Population'] = yearly_data['Population_Pred']

# Verify rows after handling missing values
print(f"Number of rows after handling missing values: {len(yearly_data)}")

# Step 6: Select the features for prediction (add 'Population' to the feature list)
features = ['mean_temp', 'mean_humidity', 'mean_windspeed', 'enso', 'mean_solar_radiation',
            'mean_wind_direction', 'rainfall', 'Population']  # Include Population column

# Step 7: Normalize the features using the scaler fitted on the training data
# Debug: Check if population is properly scaled
print("Before Scaling - Population Feature Example:")
print(yearly_data['Population'].head())

# Proper scaling of the population feature is essential
scaler = StandardScaler()
X_yearly = yearly_data[features].values

# Apply scaling only to non-population features to ensure population remains impactful
X_scaled = scaler.fit_transform(X_yearly[:, :-1])  # Exclude Population
X_yearly_scaled = np.column_stack((X_scaled, yearly_data['Population'].values))  # Add unscaled Population

# Debug: After Scaling
print("After Scaling - Example Feature Values:")
print(X_yearly_scaled[:5])

# Step 8: Convert the yearly features into PyTorch tensors
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X_yearly_tensor = torch.tensor(X_yearly_scaled, dtype=torch.float32).to(device)

# Step 9: Load the trained model
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegressionModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, 256)  # Increased units
        self.layer2 = nn.Linear(256, 128)  # Added complexity
        self.layer3 = nn.Linear(128, 64)
        self.layer4 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(p=0.3)  # Increased dropout

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.dropout(x)
        x = torch.relu(self.layer2(x))
        x = self.dropout(x)
        x = torch.relu(self.layer3(x))
        return self.layer4(x)

# Initialize the model (ensure the same architecture as the trained model)
model = LinearRegressionModel(X_yearly_tensor.shape[1]).to(device)

# Load trained model weights (Replace 'model_path' with actual path where the model is saved)
model.load_state_dict(torch.load('../data/NSW/saved_model.pth'))  # Replace with your saved model path
model.eval()

# Step 10: Predict yearly demand using the model
with torch.no_grad():
    yearly_demand_predictions = model(X_yearly_tensor)

# Step 11: Convert predictions to NumPy array and print yearly demand predictions
yearly_demand_predictions = yearly_demand_predictions.cpu().numpy()

# Merge predictions with the year and population forecast data for better visualization
yearly_results = yearly_data[['Year', 'Population']].copy()
yearly_results['Predicted_Demand'] = yearly_demand_predictions

# Print or save the results
print(yearly_results)

# Optionally save results to a CSV file
yearly_results.to_csv('../data/NSW/predicted_yearly_demand.csv', index=False)

# Step 12: Plot the actual population and predicted demand
plt.figure(figsize=(12, 6))

# Plot Population Forecast
plt.subplot(1, 2, 1)
plt.plot(yearly_results['Year'], yearly_results['Population'], label='Population', color='blue', marker='o')
plt.title('Population Forecast')
plt.xlabel('Year')
plt.ylabel('Population')
plt.grid(True)
plt.legend()

# Plot Predicted Energy Demand
plt.subplot(1, 2, 2)
plt.plot(yearly_results['Year'], yearly_results['Predicted_Demand'], label='Predicted Demand', color='orange', marker='o')
plt.title('Predicted Energy Demand')
plt.xlabel('Year')
plt.ylabel('Demand')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

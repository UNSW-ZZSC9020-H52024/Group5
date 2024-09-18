import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import os

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
    'mean_temp', 'SOI','SST_DIFF','mean_humidity', 'mean_windspeed',  'mean_solar_radiation',
    'mean_wind_direction', 'rainfall', 'Population', 'DAYOFWEEK', 'DAYOFYEAR'
]

# Step 2: Prepare data for PyTorch model training
def prepare_forecast_data(population_forecast, features):
    X = population_forecast[features].values
    return X

# Prepare daily data
X_daily = prepare_forecast_data(population_forecast_daily, features)

# Step 3: Normalize the data
scaler = StandardScaler()
X_daily = scaler.fit_transform(X_daily)

# Step 4: Move data to GPU or CPU using PyTorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Convert data to tensors and move to the chosen device (CPU or GPU)
X_daily_tensor = torch.tensor(X_daily, dtype=torch.float32).to(device)

# Step 5: Define the PyTorch Linear Regression Model
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

# Step 6: Load the saved Linear Regression model
model = LinearRegressionModel(X_daily_tensor.shape[1]).to(device)
model_path = '../data/NSW/saved_model.pth'  # Define where your model is saved
model.load_state_dict(torch.load(model_path))
model.eval()

# Step 7: Make predictions for daily data
with torch.no_grad():
    y_pred_daily = model(X_daily_tensor).cpu().numpy().flatten()

# Assign predicted values to DataFrame
population_forecast_daily['Predicted_Demand'] = y_pred_daily

# Step 8: Aggregate the daily predictions to weekly, monthly, and yearly
population_forecast_weekly = population_forecast_daily.groupby(['Year', 'Week'], as_index=False).agg({'Predicted_Demand': 'sum'})
population_forecast_monthly = population_forecast_daily.groupby(['Year', 'Month'], as_index=False).agg({'Predicted_Demand': 'sum'})
population_forecast_yearly = population_forecast_daily.groupby(['Year'], as_index=False).agg({'Predicted_Demand': 'sum'})

# Create a date column for weekly data
population_forecast_weekly['Date'] = population_forecast_weekly.apply(lambda row: pd.to_datetime(f'{int(row["Year"])}-{int(row["Week"])}-1', format='%Y-%W-%w'), axis=1)

# Create a date column for monthly data
population_forecast_monthly['Date'] = population_forecast_monthly.apply(lambda row: pd.to_datetime(f'{int(row["Year"])}-{int(row["Month"])}-01'), axis=1)

# Use the first day of the year for yearly data
population_forecast_yearly['Date'] = population_forecast_yearly['Year'].apply(lambda year: pd.to_datetime(f'{int(year)}-01-01'))

# Step 9: Save the forecast results to CSV
population_forecast_daily.to_csv('../data/NSW/daily_demand_forecast_LR.csv', index=False)
population_forecast_weekly.to_csv('../data/NSW/weekly_demand_forecast_LR.csv', index=False)
population_forecast_monthly.to_csv('../data/NSW/monthly_demand_forecast_LR.csv', index=False)
population_forecast_yearly.to_csv('../data/NSW/yearly_demand_forecast_LR.csv', index=False)

print("Demand forecasts saved successfully to CSV files.")

# Step 10: Visualization - Subplots for Daily, Weekly, and Monthly Forecasts

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

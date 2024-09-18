# -*- coding: utf-8 -*-
"""
Energy Demand Prediction Model (Linear Regression with PyTorch, Tuned)

Author: Manoj
"""

import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score  # For RMSE and R²
import os

# Step 1: Load Preprocessed Data
data_path = '../data/NSW/data_for_ml.csv'  # Update the path based on your file
preprocessed_data = pd.read_csv(data_path)


preprocessed_data.head()

# Step 2: Check for missing or infinite values in the dataset
def check_data_issues(data, columns):
    for col in columns:
        print(f"Checking column: {col}")
        missing_vals = data[col].isnull().sum()
        infinite_vals = np.isinf(data[col]).sum()
        print(f"Missing values: {missing_vals}, Infinite values: {infinite_vals}")
        if missing_vals > 0 or infinite_vals > 0:
            print(f"Column '{col}' contains issues and should be handled.")

columns_to_check = [
    'mean_temp', 'mean_humidity', 'mean_windspeed', 'TOTALDEMAND',
    'enso', 'mean_solar_radiation', 'mean_wind_direction', 'rainfall', 'Population','DAYOFWEEK','DAYOFYEAR'
]
check_data_issues(preprocessed_data, columns_to_check)

# Step 3: Prepare data for PyTorch model training
# Use all relevant features for prediction
features = [
    'mean_temp', 'mean_humidity', 'mean_windspeed', 'enso', 'mean_solar_radiation',
    'mean_wind_direction', 'rainfall', 'Population','DAYOFWEEK','DAYOFYEAR'
]
X = preprocessed_data[features].values
y = preprocessed_data['TOTALDEMAND'].values

# Step 4: Normalize the data to stabilize training
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Step 5: Split the data into training and test sets
split_index = int(len(X) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Step 6: Move data to GPU or CPU using PyTorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Convert data to tensors and move to the chosen device (CPU or GPU)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

# Step 7: Define the PyTorch Linear Regression Model with Dropout and extra layers
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

# Step 8: Initialize model, optimizer, and loss function with weight decay (L2 regularization)
model = LinearRegressionModel(X_train_tensor.shape[1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # Add L2 regularization
loss_fn = nn.MSELoss()

# Step 9: Initialize weights using Xavier initialization
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)

model.apply(init_weights)

# Step 10: Train the model with increased epochs, mini-batch gradient descent, and scheduler
def train_model(model, X_train, y_train, optimizer, loss_fn, batch_size=64, epochs=500):
    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)  # Reduce learning rate every 50 epochs
    
    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            y_pred = model(batch_X)
            loss = loss_fn(y_pred, batch_y)
            loss.backward()
            optimizer.step()
        scheduler.step()  # Update the learning rate
        if epoch % 10 == 0:
            print(f'Epoch {epoch}/{epochs}, Loss: {loss.item()}')

train_model(model, X_train_tensor, y_train_tensor, optimizer, loss_fn, epochs=500)

# Step 11: Save the model
model_save_path = '../data/NSW/saved_model.pth'  # Define where to save the model
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# Step 12: Evaluate the model
model.eval()
with torch.no_grad():
    y_pred_train = model(X_train_tensor)
    y_pred_test = model(X_test_tensor)

    # Calculate Train and Test MSE
    train_mse = loss_fn(y_pred_train, y_train_tensor).item()
    test_mse = loss_fn(y_pred_test, y_test_tensor).item()
    
    # Calculate RMSE (Root Mean Squared Error)
    train_rmse = np.sqrt(train_mse)
    test_rmse = np.sqrt(test_mse)
    
    # Convert predictions and actual values to CPU numpy arrays
    y_train_cpu = y_train_tensor.cpu().numpy()
    y_test_cpu = y_test_tensor.cpu().numpy()
    y_pred_train_cpu = y_pred_train.cpu().numpy()
    y_pred_test_cpu = y_pred_test.cpu().numpy()
    
    # Calculate R² (R-squared)
    train_r2 = r2_score(y_train_cpu, y_pred_train_cpu)
    test_r2 = r2_score(y_test_cpu, y_pred_test_cpu)

    # Print evaluation metrics
    print(f'Train MSE: {train_mse}, Train RMSE: {train_rmse}, Train R²: {train_r2}')
    print(f'Test MSE: {test_mse}, Test RMSE: {test_rmse}, Test R²: {test_r2}')

# Step 13: Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(y_test_tensor.cpu().numpy()[:100], label='Actual')
plt.plot(y_pred_test.cpu().numpy()[:100], label='Predicted')
plt.title('Actual vs Predicted Energy Demand (Tuned Linear Regression with PyTorch)')
plt.legend()
plt.show()

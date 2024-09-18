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
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load Preprocessed Data
def load_and_aggregate_data():
    data_path = '../data/NSW/data_for_ml.csv'  # Update the path based on your file
    daily_data = pd.read_csv(data_path)
    daily_data['Date'] = pd.to_datetime(daily_data['Date'])
    daily_data['Year'] = daily_data['Date'].dt.year
    daily_data['Month'] = daily_data['Date'].dt.month
    daily_data['Week'] = daily_data['Date'].dt.isocalendar().week
    return daily_data

# Step 2: Prepare data for PyTorch model training
def prepare_data_for_training(data):
    features = [
        'mean_temp', 'mean_humidity', 'mean_windspeed', 'enso', 'mean_solar_radiation',
        'mean_wind_direction', 'rainfall', 'Population', 'DAYOFWEEK', 'DAYOFYEAR'
    ]
    X = data[features].values
    y = data['TOTALDEMAND'].values

    # Normalize the data to stabilize training
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split the data into training and test sets
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    return X_train, X_test, y_train, y_test, scaler

# Step 3: Move data to GPU or CPU using PyTorch
def move_data_to_device(X_train, X_test, y_train, y_test):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Convert data to tensors and move to the chosen device (CPU or GPU)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, device

# Step 4: Define the PyTorch Linear Regression Model with Dropout and extra layers
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

# Step 5: Initialize model, optimizer, and loss function with weight decay (L2 regularization)
def init_model(input_dim, device):
    model = LinearRegressionModel(input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # Add L2 regularization
    loss_fn = nn.MSELoss()

    # Initialize weights using Xavier initialization
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
    model.apply(init_weights)

    return model, optimizer, loss_fn

# Step 6: Train the model
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

# Step 7: Evaluate the model and plot the results
def evaluate_and_plot(model, X_test_tensor, y_test_tensor, daily_data, scaler, device):
    model.eval()
    with torch.no_grad():
        y_pred_test = model(X_test_tensor).cpu().numpy()

        # Predict on full dataset
        X_full = scaler.transform(daily_data[['mean_temp', 'mean_humidity', 'mean_windspeed',
                                              'enso', 'mean_solar_radiation', 'mean_wind_direction',
                                              'rainfall', 'Population', 'DAYOFWEEK', 'DAYOFYEAR']].values)
        X_full_tensor = torch.tensor(X_full, dtype=torch.float32).to(device)
        y_pred_full = model(X_full_tensor).cpu().numpy()

    # Plotting the results
    plt.figure(figsize=(12, 6))
    plt.plot(daily_data['Date'], daily_data['TOTALDEMAND'], label='Actual Demand', color='blue')
    plt.plot(daily_data['Date'], y_pred_full, label='Predicted Demand', color='red', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Demand')
    plt.title('Daily Energy Demand vs. Predicted Demand (Linear Regression)')
    plt.legend()
    plt.show()

# Step 8: Main function
def main():
    # Load and preprocess the data
    daily_data = load_and_aggregate_data()

    # Prepare data for training
    X_train, X_test, y_train, y_test, scaler = prepare_data_for_training(daily_data)
    X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, device = move_data_to_device(X_train, X_test, y_train, y_test)

    # Initialize and train the model
    model, optimizer, loss_fn = init_model(X_train_tensor.shape[1], device)
    train_model(model, X_train_tensor, y_train_tensor, optimizer, loss_fn, epochs=500)

    # Evaluate the model and plot the results
    evaluate_and_plot(model, X_test_tensor, y_test_tensor, daily_data, scaler, device)

if __name__ == '__main__':
    main()

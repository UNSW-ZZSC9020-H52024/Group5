# -*- coding: utf-8 -*-
"""
Energy Demand Prediction Model Validation (MLPRegressor with scikit-learn)

Author: Manoj
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import pickle

# Step 1: Load Preprocessed Data and Aggregate
def load_and_aggregate_data():
    data_path = '../data/NSW/data_for_ml.csv'  # Update the path based on your file
    daily_data = pd.read_csv(data_path)
    daily_data['Date'] = pd.to_datetime(daily_data['Date'])
    daily_data['Year'] = daily_data['Date'].dt.year
    daily_data['Month'] = daily_data['Date'].dt.month
    daily_data['Week'] = daily_data['Date'].dt.isocalendar().week
    return daily_data

# Step 2: Prepare Data for Model Validation
def prepare_data_for_training(data):
    features = [
        'mean_temp', 'SOI', 'SST_DIFF', 'mean_humidity', 'mean_windspeed', 'mean_solar_radiation',
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

# Step 3: Load the saved MLP model
def load_model(model_path='../data/NSW/saved_mlp_model.pkl'):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded from {model_path}")
    return model

# Step 4: Evaluate the Model
def evaluate_model(model, X_train, y_train, X_test, y_test):
    # Make predictions on training and test sets
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Calculate evaluation metrics
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_rmse = np.sqrt(train_mse)
    test_rmse = np.sqrt(test_mse)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)

    # Print evaluation metrics
    print(f'Train MSE: {train_mse}, Train RMSE: {train_rmse}, Train R²: {train_r2}')
    print(f'Test MSE: {test_mse}, Test RMSE: {test_rmse}, Test R²: {test_r2}')

    return y_pred_train, y_pred_test

# Step 5: Plot Actual vs Predicted
def plot_results(daily_data, model, scaler):
    # Predict on the full dataset
    features = [
        'mean_temp', 'SOI', 'SST_DIFF', 'mean_humidity', 'mean_windspeed', 'mean_solar_radiation',
        'mean_wind_direction', 'rainfall', 'Population', 'DAYOFWEEK', 'DAYOFYEAR'
    ]
    X_full = scaler.transform(daily_data[features].values)
    y_pred_full = model.predict(X_full)

    # Plotting the results
    plt.figure(figsize=(12, 6))
    plt.plot(daily_data['Date'], daily_data['TOTALDEMAND'], label='Actual Demand', color='blue')
    plt.plot(daily_data['Date'], y_pred_full, label='Predicted Demand', color='red', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Demand')
    plt.title('Daily Energy Demand vs. Predicted Demand (MLPRegressor)')
    plt.legend()
    plt.show()

# Step 6: Main function to validate the model
def main():
    # Load and preprocess the data
    daily_data = load_and_aggregate_data()

    # Prepare data for training and testing
    X_train, X_test, y_train, y_test, scaler = prepare_data_for_training(daily_data)

    # Load the trained model
    model = load_model()

    # Evaluate the model
    evaluate_model(model, X_train, y_train, X_test, y_test)

    # Plot the results
    plot_results(daily_data, model, scaler)

if __name__ == '__main__':
    main()

# Adjusted MLPRegressor model training code

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

# Load Preprocessed Data
data_path = '../data/NSW/data_for_ml.csv'
preprocessed_data = pd.read_csv(data_path)

# Check columns for missing values or infinite values
def check_data_issues(data, columns):
    for col in columns:
        missing_vals = data[col].isnull().sum()
        infinite_vals = np.isinf(data[col]).sum()
        print(f"Checking column: {col}")
        print(f"Missing values: {missing_vals}, Infinite values: {infinite_vals}")

columns_to_check = [
    'mean_temp', 'SOI', 'SST_DIFF', 'mean_humidity', 'mean_windspeed', 'TOTALDEMAND',
    'mean_solar_radiation', 'mean_wind_direction', 'rainfall', 'Population', 'DAYOFWEEK', 'DAYOFYEAR'
]
check_data_issues(preprocessed_data, columns_to_check)

# Prepare features and target variable
features = [
    'mean_temp', 'SOI', 'SST_DIFF', 'mean_humidity', 'mean_windspeed', 'mean_solar_radiation',
    'mean_wind_direction', 'rainfall', 'Population', 'DAYOFWEEK', 'DAYOFYEAR'
]
X = preprocessed_data[features].values
y = preprocessed_data['TOTALDEMAND'].values

# Normalize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into training and testing sets
split_index = int(len(X) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Define and train the adjusted MLPRegressor model
model = MLPRegressor(
    solver='adam',              # Changed to 'adam' for better convergence
    max_iter=2000,              # Increased max iterations
    hidden_layer_sizes=(100, 50),  # Adjusted hidden layer sizes
    learning_rate_init=0.001,   # Lower initial learning rate
    activation='relu',          # Changed activation function to ReLU
    random_state=42
)
model.fit(X_train, y_train)

# Save the model
model_save_path = '../data/NSW/saved_mlp_model.pkl'
with open(model_save_path, 'wb') as f:
    pickle.dump(model, f)
print(f"Model saved to {model_save_path}")

# Evaluate the model
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Calculate metrics
train_mse = mean_squared_error(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)
train_rmse = np.sqrt(train_mse)
test_rmse = np.sqrt(test_mse)
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

# Print metrics
print(f'Train MSE: {train_mse}, Train RMSE: {train_rmse}, Train R²: {train_r2}')
print(f'Test MSE: {test_mse}, Test RMSE: {test_rmse}, Test R²: {test_r2}')

# Plot actual vs predicted
plt.figure(figsize=(10, 6))
plt.plot(y_test[:100], label='Actual')
plt.plot(y_pred_test[:100], label='Predicted')
plt.title('Actual vs Predicted Energy Demand (MLPRegressor)')
plt.legend()
plt.show()

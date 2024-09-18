import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import torch
import pickle

# Step 1: Load Preprocessed Data
data_path = '../data/NSW/data_for_ml.csv'  # Update the path based on your file
preprocessed_data = pd.read_csv(data_path)

# Step 2: Prepare data
features = [
    'mean_temp', 'mean_humidity', 'mean_windspeed', 'enso', 'mean_solar_radiation',
    'mean_wind_direction', 'rainfall', 'Population','DAYOFWEEK','DAYOFYEAR'
]
X = preprocessed_data[features].values
y = preprocessed_data['TOTALDEMAND'].values

# Step 3: Normalize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Step 4: Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Define the Random Forest model
rf = RandomForestRegressor(random_state=42)

# Step 6: Define hyperparameter grid
param_distributions = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [1.0, 'sqrt', 'log2']  # Remove 'auto', use 1.0 as default or other options
}


# Step 7: Perform RandomizedSearchCV to find the best parameters
random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_distributions, 
                                   n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)

# Fit the model
random_search.fit(X_train, y_train)

# Step 8: Get the best parameters
best_params = random_search.best_params_
print(f"Best Parameters: {best_params}")

# Step 9: Train the Random Forest model with the best parameters
best_rf = random_search.best_estimator_

# Step 10: Make predictions and evaluate the model
y_pred_train = best_rf.predict(X_train)
y_pred_test = best_rf.predict(X_test)

# Calculate Train and Test RMSE and R²
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

# Print the results
print(f"Train RMSE: {train_rmse}, Train R²: {train_r2}")
print(f"Test RMSE: {test_rmse}, Test R²: {test_r2}")

# Step 11: Save the tuned model
model_save_path = '../data/NSW/saved_tuned_rf_model.pkl'
with open(model_save_path, 'wb') as f:
    pickle.dump(best_rf, f)
print(f"Tuned Random Forest model saved to {model_save_path}")

# Step 12: Plotting the results
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(y_test[:100], label='Actual')
plt.plot(y_pred_test[:100], label='Predicted')
plt.title('Actual vs Predicted Energy Demand (Tuned Random Forest)')
plt.legend()
plt.show()

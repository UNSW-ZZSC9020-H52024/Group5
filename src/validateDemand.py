# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 22:45:26 2024

@author: Manoj
"""

import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load the predicted demand data from 2023 onward
predicted_data_path = '../data/NSW/predicted_yearly_demand.csv'  # Replace with actual path
predicted_data = pd.read_csv(predicted_data_path)

# Step 2: Load the historical total demand data (up to 2021)
historical_data_path = '../data/NSW/totaldemand_nsw.csv'  # Replace with actual path
historical_data = pd.read_csv(historical_data_path)

# Step 3: Extract 'Year' from 'DATETIME' in the historical data and sum the total demand by year
historical_data['DATETIME'] = pd.to_datetime(historical_data['DATETIME'])
historical_data['Year'] = historical_data['DATETIME'].dt.year
historical_yearly_demand = historical_data.groupby('Year')['TOTALDEMAND'].sum().reset_index()

# Step 4: Merge the historical and predicted demand data
# Filter the historical data up to 2021
historical_yearly_demand = historical_yearly_demand[historical_yearly_demand['Year'] <= 2021]

# Combine historical and predicted data into a single DataFrame
combined_data = pd.concat([
    historical_yearly_demand[['Year', 'TOTALDEMAND']].rename(columns={'TOTALDEMAND': 'Demand'}),
    predicted_data[['Year', 'Predicted_Demand']].rename(columns={'Predicted_Demand': 'Demand'})
], ignore_index=True)

# Step 5: Generate a combined CSV file for the yearly demand
combined_data.to_csv('../data/NSW/combined_yearly_demand.csv', index=False)
print("Combined yearly demand CSV has been generated.")

# Step 6: Plot the bar chart for historical and predicted demand using dual y-axes
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot historical demand on the left y-axis
ax1.bar(historical_yearly_demand['Year'], historical_yearly_demand['TOTALDEMAND'], 
        color='blue', label='Historical Demand', alpha=0.7)
ax1.set_xlabel('Year')
ax1.set_ylabel('Historical Energy Demand', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_title('Historical and Predicted Energy Demand')

# Create a second y-axis for the predicted demand
ax2 = ax1.twinx()
ax2.bar(predicted_data['Year'], predicted_data['Predicted_Demand'], 
        color='orange', label='Predicted Demand', alpha=0.7)
ax2.set_ylabel('Predicted Energy Demand', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')

# Add legends for both plots
fig.tight_layout()
plt.show()

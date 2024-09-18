# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = '../data/NSW/popforecast_nsw.csv'
population_data = pd.read_csv(file_path, skiprows=1)  # Skip the extra header row

# Rename the columns
population_data.columns = ['Year', 'High series', 'Medium series', 'Low series', 
                           'Smallest population', 'Largest population']

# Remove any rows with missing data
population_data = population_data.dropna()

# Convert 'Year' column to integer and 'Medium series' to numeric after removing commas
population_data['Year'] = population_data['Year'].astype(int)
population_data['Medium series'] = pd.to_numeric(population_data['Medium series'].str.replace(',', ''))

# Set the start date for each year as January 1
population_data['Start Date'] = pd.to_datetime(population_data['Year'].astype(str) + '-01-01')

# Set the Start Date as the index for interpolation
population_data.set_index('Start Date', inplace=True)

# Fill missing values before interpolation to avoid gaps
population_data['Medium series'].ffill(inplace=True)
population_data['Medium series'].bfill(inplace=True)

# Linear interpolation for daily, weekly, monthly, and yearly intervals
# Resample and interpolate for each time frequency, using the correct codes

# Daily Interpolation
population_data_daily = population_data['Medium series'].resample('D').interpolate('linear')

# Weekly Interpolation
population_data_weekly = population_data['Medium series'].resample('W').interpolate('linear')

# Monthly Interpolation (using 'MS' for month start to ensure full coverage)
population_data_monthly = population_data['Medium series'].resample('MS').interpolate('linear').ffill().bfill()

# Yearly Interpolation (using 'YS' for year start to ensure full yearly coverage)
population_data_yearly = population_data['Medium series'].resample('YS').interpolate('linear').ffill().bfill()

# Aggregate the daily, weekly, and monthly interpolations to yearly averages
population_data_daily_yearly = population_data_daily.resample('YS').mean()
population_data_weekly_yearly = population_data_weekly.resample('YS').mean()
population_data_monthly_yearly = population_data_monthly.resample('YS').mean()

# Create DataFrames with dates and populations for CSV export
population_data_daily_df = pd.DataFrame({'Date': population_data_daily.index, 'Population': population_data_daily.values})
population_data_weekly_df = pd.DataFrame({'Date': population_data_weekly.index, 'Population': population_data_weekly.values})
population_data_monthly_df = pd.DataFrame({'Date': population_data_monthly.index, 'Population': population_data_monthly.values})
population_data_yearly_df = pd.DataFrame({'Date': population_data_yearly.index, 'Population': population_data_yearly.values})

# Save the forecasts to CSV files
population_data_daily_df.to_csv('../data/Population/population_forecast_linear_daily.csv', index=False)
population_data_weekly_df.to_csv('../data/Population/population_forecast_linear_weekly.csv', index=False)
population_data_monthly_df.to_csv('../data/Population/population_forecast_linear_monthly.csv', index=False)
population_data_yearly_df.to_csv('../data/Population/population_forecast_linear_yearly.csv', index=False)

# Create DataFrames for aggregated yearly means
aggregated_yearly_df = pd.DataFrame({
    'Date': population_data_yearly.index,
    'Aggregated Daily': population_data_daily_yearly.values,
    'Aggregated Weekly': population_data_weekly_yearly.values,
    'Aggregated Monthly': population_data_monthly_yearly.values,
    'Yearly Interpolation': population_data_yearly.values
})

# Save the aggregated yearly data to CSV files
aggregated_yearly_df.to_csv('../data/Population/population_aggregated_yearly.csv', index=False)

# Plot the comparison between aggregated values and yearly interpolation
plt.figure(figsize=(12, 8))

# Plot the aggregated daily, weekly, and monthly against the yearly interpolation
plt.plot(aggregated_yearly_df['Date'], aggregated_yearly_df['Aggregated Daily'], label='Aggregated Daily', linestyle='--', color='blue', linewidth=1.5)
plt.plot(aggregated_yearly_df['Date'], aggregated_yearly_df['Aggregated Weekly'], label='Aggregated Weekly', linestyle='-.', color='green', linewidth=1.5)
plt.plot(aggregated_yearly_df['Date'], aggregated_yearly_df['Aggregated Monthly'], label='Aggregated Monthly', linestyle=':', color='purple', linewidth=2)
plt.plot(aggregated_yearly_df['Date'], aggregated_yearly_df['Yearly Interpolation'], label='Yearly Interpolation', linestyle='-', color='red', linewidth=2.5)

# Add labels and title
plt.xlabel('Date')
plt.ylabel('Population')
plt.title('Validation of Yearly Forecast: Aggregated Daily, Weekly, Monthly vs Yearly Interpolation')

# Add a legend
plt.legend(loc='best')

# Show the plot
plt.grid(True)
plt.show()

# Display a success message
print('Validation completed. Aggregated yearly means have been compared with yearly interpolation.')

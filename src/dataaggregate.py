import pandas as pd

def fix_24_hour_time(df):
    # Find rows where Time is "24:00"
    mask = df['Time'] == "24:00"
    
    # For those rows, set the Time to "00:00"
    df.loc[mask, 'Time'] = "00:00"
    
    # Convert 'Date' to datetime format before adding one day
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    
    # Increment the Date by one day for rows where Time was "24:00"
    df.loc[mask, 'Date'] = df.loc[mask, 'Date'] + pd.DateOffset(days=1)
    
    # Convert the 'Date' column back to the original format if necessary
    df['Date'] = df['Date'].dt.strftime('%d/%m/%Y')
    
    return df

def aggregate_temperature_data(input_csv, output_csv):
    # Load the merged CSV file
    df = pd.read_csv(input_csv)

    # Ensure 'Date' and 'Time' columns are present
    if 'Date' not in df.columns or 'Time' not in df.columns:
        raise ValueError("The input CSV must contain 'Date' and 'Time' columns.")

    # Fix any instances where Time is "24:00"
    df = fix_24_hour_time(df)

    # Select only the temperature columns (assuming temperature columns have 'TEMP' in their name)
    temp_columns = [col for col in df.columns if 'TEMP' in col]

    # Calculate mean and median across all temperature columns for each row
    df['mean_temp'] = df[temp_columns].mean(axis=1)
    df['median_temp'] = df[temp_columns].median(axis=1)

    # Group by 'Date' and 'Time' to calculate mean and median for unique combinations of Date and Time
    aggregated_df = df.groupby(['Date', 'Time']).agg({
        'mean_temp': 'mean',
        'median_temp': 'median'
    }).reset_index()

    # Save the result to a new CSV
    aggregated_df.to_csv(output_csv, index=False)

    print(f'Aggregated data saved to {output_csv}')


# Path to the merged CSV file
input_csv = 'C:/Users/Manoj/Documents/GitHub/ZZSC9020 H52024/Group5_1/src/merged_temperature_data.csv'

# Path to save the aggregated CSV
output_csv = 'C:/Users/Manoj/Documents/GitHub/ZZSC9020 H52024/Group5_1/src/aggregated_temperature_data.csv'

# Call the function to aggregate the temperature data
aggregate_temperature_data(input_csv, output_csv)

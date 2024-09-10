import os
import pandas as pd

def merge_rainfall_data(folder_path):
    # Initialize an empty DataFrame to store the merged data
    merged_df = pd.DataFrame()

    # Iterate through all files in the folder
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".csv"):
                # Load the CSV file with a specified encoding
                file_path = os.path.join(root, file)
                
                try:
                    # Use ISO-8859-1 (Latin-1) encoding to avoid decoding errors
                    temp_df = pd.read_csv(file_path, encoding='ISO-8859-1')

                    # Extract the location name from the file name (or define it as you prefer)
                    location_name = os.path.splitext(file)[0]

                    # Remove unwanted rows and reset the index
                    temp_df.columns = temp_df.iloc[1]  # Set second row as headers
                    temp_df = temp_df.drop([0, 1]).reset_index(drop=True)  # Remove first two rows

                    # Rename columns for rainfall data (assuming the third column holds the rainfall data)
                    rainfall_column_name = temp_df.columns[2]  # Keep the original rainfall column name
                    temp_df = temp_df.rename(columns={temp_df.columns[0]: "Date", temp_df.columns[1]: "Time", temp_df.columns[2]: f"{rainfall_column_name}_{location_name}"})

                    # If merged_df is empty, initialize it with Date and Time
                    if merged_df.empty:
                        merged_df = temp_df[['Date', 'Time', f'{rainfall_column_name}_{location_name}']]
                    else:
                        # Check if the column for the location already exists in the merged_df
                        if f'{rainfall_column_name}_{location_name}' not in merged_df.columns:
                            # Merge on Date and Time if the location does not exist
                            merged_df = pd.merge(merged_df, temp_df[['Date', 'Time', f'{rainfall_column_name}_{location_name}']], on=['Date', 'Time'], how='outer')
                        else:
                            # If the location already exists, append only the new data
                            merged_df.update(temp_df.set_index(['Date', 'Time']))

                except Exception as e:
                    print(f'Error reading file {file}: {e}')
    
    return merged_df

# Path to the folder containing the rainfall CSV files
folder_path = 'C:/Users/Manoj/Documents/GitHub/ZZSC9020 H52024/Group5_1/data/NSW/Hourly_weather_data/Rainfall\csv'

# Call the function to merge rainfall data
merged_data = merge_rainfall_data(folder_path)

# Save the merged data to a new CSV
merged_data.to_csv('merged_rainfall_data.csv', index=False)

print("Merging completed. Data saved to 'merged_rainfall_data.csv'")

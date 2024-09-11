import os
import pandas as pd

def convert_xls_to_csv(folder_path):
    # Iterate through all subfolders in the given folder
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".xls") or file.endswith(".xlsx"):
                # Construct the full file path
                xls_file_path = os.path.join(root, file)
                
                # Try reading the file using the appropriate engine
                try:
                    if file.endswith(".xls"):
                        # Load the .xls file using xlrd engine
                        df = pd.read_excel(xls_file_path, engine='xlrd')
                    elif file.endswith(".xlsx"):
                        # Load the .xlsx file using openpyxl engine
                        df = pd.read_excel(xls_file_path, engine='openpyxl')
                    
                    # Create a 'csv' subfolder in the current folder if it doesn't exist
                    csv_folder = os.path.join(root, 'csv')
                    if not os.path.exists(csv_folder):
                        os.makedirs(csv_folder)
                    
                    # Construct the csv file path in the 'csv' subfolder
                    csv_file_name = os.path.splitext(file)[0] + ".csv"
                    csv_file_path = os.path.join(csv_folder, csv_file_name)
                    
                    # Save the DataFrame as a .csv file
                    df.to_csv(csv_file_path, index=False)
                    
                    print(f'Converted {file} to CSV and saved at {csv_file_path}')
                
                except Exception as e:
                    print(f'Failed to convert {file}. Error: {e}')

# Replace 'your_directory_path' with the path to your directory containing .xls files
folder_path = 'C:/Users/Manoj/Documents/GitHub/ZZSC9020 H52024/Group5_1/data/NSW/Hourly_weather_data/WDR'
convert_xls_to_csv(folder_path)

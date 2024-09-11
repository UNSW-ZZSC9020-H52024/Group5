import pandas as pd

#Load daily_soi.txt as a pandas dataframe with the first row as the header
soi_load = pd.read_csv('data/enso/daily_soi.txt', delim_whitespace=True, header=0)

#drop the Tahiti and Darwin columns
soi_load.drop(columns=['Tahiti', 'Darwin'], inplace=True)

#join the Year and Day columns to create a new column 'date' in datetime format
soi_load['date'] = soi_load['Year'].astype(str) + ' ' + soi_load['Day'].astype(str)
soi_load['date'] = pd.to_datetime(soi_load['date'], format='%Y %j')

#create a three-month rolling average of the 'SOI' column and save it as a new column 'soi_3m_avg'
soi_load['soi_3m_avg'] = soi_load['SOI'].rolling(window=90).mean()

# Define a function to compute nino, nina, and enso
def compute_enso(row):
    nino = 1 if row > 7 else 0
    nina = -1 if row < -7 else 0
    return nino, nina, nino + nina

# Apply the function to the 'soi_3m_avg' column and create new columns
soi_load[['nino', 'nina', 'enso']] = soi_load['soi_3m_avg'].apply(lambda x: pd.Series(compute_enso(x)))

#drop the 'Year', 'Day', 'SOI', 'soi_3m_avg', 'nino', and 'nina' columns
soi_load.drop(columns=['Year', 'Day', 'soi_3m_avg', 'nino', 'nina'], inplace=True)

#delete all rows before 2009-12-31
soi_load = soi_load[soi_load['date'] >= '2009-12-31']

#save the file as daily_soi_load.csv
soi_load.to_csv('data/enso/daily_enso.csv', index=False)
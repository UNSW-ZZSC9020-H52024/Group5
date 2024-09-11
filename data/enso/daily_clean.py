#This file opens and cleans the daily_soi.txt and nino_3.4.txt files then joins them on the date
#It then creates a new column 'enso' that is 1 for el nino, -1 for la nina, and 0 for enso neutral based on the BoM criteria ()

import pandas as pd

#Load daily_soi.txt as a DataFrame
soi_load = pd.read_csv('data/enso/daily_soi.txt', delim_whitespace=True, header=0)

#drop the Tahiti and Darwin columns
soi_load.drop(columns=['Tahiti', 'Darwin'], inplace=True)

#join the Year and Day columns
soi_load['DATETIME'] = soi_load['Year'].astype(str) + ' ' + soi_load['Day'].astype(str)
soi_load['DATETIME'] = pd.to_datetime(soi_load['DATETIME'], format='%Y %j')

#load nino_3.4.txt as a pandas dataframe with the first row as the header
sst = pd.read_csv('data/enso/nino_3.4.txt', sep=',', names=['STARTDATETIME','ENDDATETIME', 'SST_DIFF'])

#convert the 'STARTDATETIME' and 'ENDDATETIME' columns to datetime format
sst['STARTDATETIME'] = pd.to_datetime(sst['STARTDATETIME'], format='%Y%m%d')
sst['ENDDATETIME'] = pd.to_datetime(sst['ENDDATETIME'], format='%Y%m%d')

sst['DATETIME'] = sst.apply(lambda x: pd.date_range(start=x['STARTDATETIME'], end=x['ENDDATETIME'], freq='D'), axis=1)

sst = sst.explode('DATETIME')

sst.drop(columns=['STARTDATETIME', 'ENDDATETIME'], inplace=True)

sst = sst.reset_index(drop=True)

#delete all rows before 2009-12-31
sst = sst[(sst['DATETIME'] >= '2009-12-31')]
sst[(sst['DATETIME'] <= '2020-12-31')]
soi_load = soi_load[(soi_load['DATETIME'] >= '2009-12-31')]
soi_load[(soi_load['DATETIME'] <= '2020-12-31')]


#merge the soi_load and sst dataframes on the 'DATETIME' column
enso = pd.merge(soi_load, sst, on='DATETIME', how='left')

#fill the missing values in the 'SST_DIFF' column with the mean of the window of the previous 14 days
enso['SST_DIFF'] = enso['SST_DIFF'].fillna(enso['SST_DIFF'].rolling(window=14, min_periods=1).mean())


#create a three-month rolling average of the 'SOI' column and save it as a new column 'soi_3m_avg'
#enso['soi_3m_avg'] = enso['SOI'].rolling(window=91).mean()
soi_load['soi_3m_avg'] = soi_load['SOI'].rolling(window=91).mean()

#la nina is when the sst_diff is greater than 0.8 and SOI is greater than 8, el nino is when the sst_diff is less than -0.8 and SOI is less than -8, and enso neutral is all other times
def compute_enso(row):
    nino = 1 if row['SST_DIFF'] > 0.8 and row['SOI'] < -8 else 0
    nina = -1 if row['SST_DIFF'] < -0.8 and row['SOI'] > 8 else 0
    return nino + nina

# Apply the function to the 'soi_3m_avg' column and create new columns
enso[['enso']] = enso.apply(lambda x: pd.Series(compute_enso(x)), axis=1)


# Define a function to compute nino, nina, and enso for soi only
def compute_enso_soi(row):
    nino = 1 if row < -8 else 0
    nina = -1 if row > 8 else 0
    return nino + nina

# Apply the function to the 'soi_3m_avg' column and create new columns
soi_load[['enso']] = soi_load['soi_3m_avg'].apply(lambda x: pd.Series(compute_enso_soi(x)))

#drop the 'Year', 'Day', 'SOI', 'soi_3m_avg', 'nino', and 'nina' columns
soi_load.drop(columns=['Year', 'Day', 'soi_3m_avg'], inplace=True)
enso.drop(columns=['Year', 'Day'], inplace=True)

#print the value counts of the 'enso' column
print("value counts of the 'enso' column (soi only first)")
print(soi_load.value_counts('enso'))
print(enso.value_counts('enso'))

#make the DATETIME column the index
soi_load.set_index('DATETIME', inplace=True)
enso.set_index('DATETIME', inplace=True)

#save the file as daily_soi_load.csv
soi_load.to_csv('data/enso/daily_enso_justSOI.csv')
enso.to_csv('data/enso/daily_enso.csv')

#Import libraries
import pandas as pd

#Load soi_monthly.txt as a pandas dataframe
soi = pd.read_csv('data/enso/soi_monthly.txt', header=None, names=['yearmonth', 'soi'])

#change the data type of 'yearmonth' to int and 'soi' to float
soi['yearmonth'] = soi['yearmonth'].astype(str)
soi['soi'] = soi['soi'].astype(float)

#only select rows where 'yearmonth' is greater than 200912
soi = soi[soi['yearmonth'] >= '200912']


#Create a new column 'year' by extracting the first 4 characters from 'yearmonth' and a second column 'month' by extracting the last 2 characters from 'yearmonth'
'''soi['year'] = soi['yearmonth'].str[:4]
soi['month'] = soi['yearmonth'].str[4:]

soi['month'] = soi['month'].astype(int)
soi['year'] = soi['year'].astype(int)

#drop the 'yearmonth' column
soi.drop(columns=['yearmonth'], inplace=True)'''


#save the file as soi_monthly.csv
soi.to_csv('data/enso/soi_monthly.csv', index=False)
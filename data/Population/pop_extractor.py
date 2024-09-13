import pandas as pd

#import the pop data that was manually changed to NSW only
pop_interim = pd.read_csv('data/Population/quarterly_pop.csv', header=0)

#change the Date column to datetime
pop_interim['Date'] = pd.to_datetime(pop_interim['Date'], format='%b-%Y')

#drop the Male and Female columns
pop_interim.drop(columns=['Male','Female'], inplace=True)

#create a new dataframe with a date range the same as pop_interim but with daily frequency
pop = pd.date_range(start='1981-06-01', end='2021-07-01', freq='D')

#left join the pop_interim dataframe with the new dataframe
pop = pd.DataFrame(pop, columns=['Date'])
pop = pd.merge(pop, pop_interim, on='Date', how='left')

#fill the missing values with the linear interpolation between non-missing values
pop['Population'] = pop['Population'].interpolate()

pop.dropna(inplace=True)

#change the Date range to the same as the other dataframes
pop = pop[pop['Date'] >= '2010-01-01']
pop = pop[pop['Date'] <= '2021-03-19']

#save the file as yearly_pop.csv
pop.to_csv('data/Population/daily_pop_nsw.csv', index=False)


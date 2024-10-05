import pandas as pd

#enso
enso = pd.read_csv('data/enso/daily_enso.csv', header=0)
enso['Date'] = pd.to_datetime(enso['DATETIME'])
enso.drop(columns=['DATETIME','enso'], inplace=True)


#humidity
hum = pd.read_csv('data/NSW/aggregated_humidity_data.csv', header=0)
hum['Date'] = pd.to_datetime(hum['Date'], dayfirst=True)
hum.drop(columns= ['Time','median_humidity'], inplace=True)
hum = hum.groupby('Date').mean()
hum = hum.sort_values(by='Date', ascending=True)

#radiation
rad = pd.read_csv('data/NSW/aggregated_solar_radiation_data.csv', header=0)
rad['Date'] = pd.to_datetime(rad['Date'], dayfirst=True)
rad.drop(columns= ['Time','median_solar_radiation'], inplace=True)
rad = rad.groupby('Date').mean()
rad = rad.sort_values(by='Date', ascending=True)

#temp
temp = pd.read_csv('data/NSW/aggregated_temperature_data.csv', header=0)
temp['Date'] = pd.to_datetime(temp['Date'], dayfirst=True)
temp.drop(columns= ['Time','median_temp'], inplace=True)
temp = temp.groupby('Date').mean()
temp = temp.sort_values(by='Date', ascending=True)

#windtheta
windtheta = pd.read_csv('data/NSW/aggregated_wind_direction_data.csv', header=0)
windtheta['Date'] = pd.to_datetime(windtheta['Date'], dayfirst=True)
windtheta.drop(columns= ['Time','median_wind_direction'], inplace=True)
windtheta = windtheta.groupby('Date').mean()
windtheta = windtheta.sort_values(by='Date', ascending=True)

#windspeed
windspeed = pd.read_csv('data/NSW/aggregated_windspeed_data.csv', header=0)
windspeed['Date'] = pd.to_datetime(windspeed['Date'], dayfirst=True)
windspeed.drop(columns= ['Time','median_windspeed'], inplace=True)
windspeed = windspeed.groupby('Date').mean()
windspeed = windspeed.sort_values(by='Date', ascending=True)

#rainfall
rain = pd.read_csv('data/NSW/median_rainfall_2010_2021.csv', header=0)
rain['Date'] = pd.to_datetime(rain['Date'])
rain.rename(columns={'Rainfall amount (millimetres)':'rainfall'}, inplace=True)
rain = rain.sort_values(by='Date', ascending=True)

#pop
pop = pd.read_csv('data/Population/daily_pop_nsw.csv', header=0)
pop['Date'] = pd.to_datetime(pop['Date'])
pop = pop.sort_values(by='Date', ascending=True)

#TOTALDEMAND
demand = pd.read_csv('data/NSW/totaldemand_nsw.csv', header=0)
demand['Date'] = pd.to_datetime(demand['DATETIME'], dayfirst=True)
demand['Date'] = demand['Date'].dt.date
demand['Date'] = pd.to_datetime(demand['Date'])
demand.drop(columns=['DATETIME'], inplace=True)
demand = demand.groupby('Date').mean()
demand = demand.sort_values(by='Date', ascending=True)


#join the dataframes
join = pd.merge(hum, enso, on='Date', how='left')
join = pd.merge(join, rad, on='Date', how='left')
join = pd.merge(join, temp, on='Date', how='left')
join = pd.merge(join, windtheta, on='Date', how='left')
join = pd.merge(join, windspeed, on='Date', how='left')
join = pd.merge(join, rain, on='Date', how='left')
join = pd.merge(join, pop, on='Date', how='left')
join = pd.merge(join, demand, on='Date', how='left')

join['DAYOFWEEK'] = join['Date'].dt.weekday
join['DAYOFYEAR'] = join['Date'].dt.dayofyear

#drop missing values
join = join.dropna()

#save as data_for_ml.csv
join.to_csv('data/NSW/data_for_ml.csv', index=False)
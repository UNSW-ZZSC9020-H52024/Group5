import pandas as pd

#import the pop data that was manually changed to NSW only
pop = pd.read_csv('data/Population/pop_his.csv', header=0)

#transpose the dataframe
pop = pop.T

#make the first row the column names
pop.columns = pop.iloc[0]

#drop the first two rows
pop = pop.drop(pop.index[0])
pop = pop.drop(pop.index[0])

#delete all null rows
pop = pop.dropna()

#delete all rows before 2009-12-31
pop = pop[(pop.index >= '2009-12-31')]

#save the file as yearly_pop_genders.csv
#pop.to_csv('data/Population/yearly_pop_genders.csv')

#drop the Male and Female columns
pop = pop.drop(columns=['Male','Female'])

#save the file as yearly_pop.csv
pop.to_csv('data/Population/yearly_pop_nsw.csv')


import pandas as pd
import numpy as np 
from sklearn import linear_model
df = pd.read_csv('data/housing_prices/train.csv')
print(df.head())

reg = linear_model.LinearRegression()
reg.fit(df[['LotArea', 'YearBuilt', 'OverallCond','YrSold']], df.SalePrice)
df2 = pd.read_csv('data/housing_prices/test.csv')

predictions = reg.predict(df2[['LotArea', 'YearBuilt', 'OverallCond','YrSold']])
print(predictions)
a = pd.DataFrame({
    'Id': df2['Id'],
    'SalePrice': predictions
})
print(a)

a.to_csv('data/housing_prices/fin.csv', index=False)

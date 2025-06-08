import pandas as pd
import numpy as np 
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures


df = pd.read_csv('data/housing_prices/train.csv')
df['YearDifference'] = df['YrSold'] - df['YearBuilt']

print(df.head())
plt.figure(figsize=(10, 6))
plt.scatter(df['YearDifference'], df['SalePrice'], alpha=0.5, color='b')
plt.show()
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(df[['LotArea', 'YearDifference', 'OverallCond', 'YrSold']])

reg = linear_model.LinearRegression()
reg.fit(X_poly, df['SalePrice'])



df2 = pd.read_csv('data/housing_prices/test.csv')
df2['YearDifference'] = df2['YrSold'] - df2['YearBuilt']

X_poly_test = poly.transform(df2[['LotArea', 'YearDifference', 'OverallCond', 'YrSold']])

predictions = reg.predict(X_poly_test)
print(predictions)
a = pd.DataFrame({
    'Id': df2['Id'],
    'SalePrice': predictions
})
print(a)

a.to_csv('data/housing_prices/fin.csv', index=False)

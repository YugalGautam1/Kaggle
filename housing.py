import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures


df = pd.read_csv('data/housing_prices/train.csv')
df['YearDifference'] = df['YrSold'] - df['YearBuilt']
print(df['SaleType'].unique())
df['Street'] = df['Street'].replace('Grvl', 0)
df['Street'] = df['Street'].replace('Pave', 1)

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(df[['LotArea', 'YearDifference', 'OverallCond', 'YrSold','OverallQual','MSSubClass','Street']])


reg = linear_model.LinearRegression()
reg.fit(X_poly, df['SalePrice'])



df2 = pd.read_csv('data/housing_prices/test.csv')
df2['YearDifference'] = df2['YrSold'] - df2['YearBuilt']
df2['Street'] = df2['Street'].replace('Grvl', 0)
df2['Street'] = df2['Street'].replace('Pave', 1)
X_poly_test = poly.transform(df2[['LotArea', 'YearDifference', 'OverallCond', 'YrSold','OverallQual','MSSubClass','Street']])

predictions = reg.predict(X_poly_test)
print(predictions)
a = pd.DataFrame({
    'Id': df2['Id'],
    'SalePrice': predictions
})

a.to_csv('data/housing_prices/fin.csv', index=False)

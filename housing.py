import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

df = pd.read_csv('data/housing_prices/train.csv')
df2 = pd.read_csv('data/housing_prices/test.csv')



df['YearDifference'] = df['YrSold'] - df['YearBuilt']
df2['YearDifference'] = df2['YrSold'] - df2['YearBuilt']


df2['YearDifference'] = df2['YrSold'] - df2['YearBuilt']
features = ['Neighborhood','BldgType','Foundation','SaleType','SaleCondition','MSSubClass','LotArea','OverallCond','OverallQual','YearBuilt','YearDifference']

df = df[['Id']+features + ['SalePrice']]
df2 = df2[['Id']+features]

df = pd.get_dummies(df, columns=['Neighborhood','BldgType','Foundation','SaleType','SaleCondition'])
df2 = pd.get_dummies(df2, columns=['Neighborhood','BldgType','Foundation','SaleType','SaleCondition'])

X_train = df.drop('SalePrice', axis=1)
y_train = df['SalePrice']
X_test = df2

gb_model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
)
gb_model.fit(X_train, y_train)

gb_preds = gb_model.predict(X_test)

output = pd.DataFrame({
    'Id': df2['Id'],
    'SalePrice': gb_preds
})
output.to_csv('data/housing_prices/fin.csv', index=False)

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

df = pd.read_csv('data/housing_prices/train.csv')
df['YearDifference'] = df['YrSold'] - df['YearBuilt']
df['Street'] = df['Street'].replace({'Grvl': 0, 'Pave': 1})

features = ['LotArea', 'OverallCond', 'YrSold', 'OverallQual', 'MSSubClass', 'Street','SaleType']

df2 = pd.read_csv('data/housing_prices/test.csv')
df2['YearDifference'] = df2['YrSold'] - df2['YearBuilt']
df2['Street'] = df2['Street'].replace({'Grvl': 0, 'Pave': 1})
sale_type_map = {
    'WD': 0,     
    'CWD': 1,    
    'VWD': 2,    
    'New': 3,    
    'COD': 4,   
    'Con': 5,    
    'ConLw': 6,  
    'ConLI': 7,  
    'ConLD': 8, 
    'Oth': 9,
}
df['SaleType'] = df['SaleType'].map(sale_type_map).fillna(9)
df2['SaleType'] = df2['SaleType'].map(sale_type_map).fillna(9)


X_train = df[features]
y_train = df['SalePrice']

X_test = df2[features]

gb_model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
)
gb_model.fit(X_train, y_train)

gb_preds = gb_model.predict(X_test)

pd.DataFrame({
    'Id': df2['Id'],
    'SalePrice': gb_preds
}).to_csv('data/housing_prices/fin.csv', index=False)
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv("C:/Users/PRADEEPHP/Music/HousePrice/train.csv")
y=data['SalePrice']
##null_columns = data.isnull().sum(axis = 0)
##null_columns=null_columns[null_columns>0]

data['LotFrontage'] = data['LotFrontage'].fillna(data['LotFrontage'].mean())
data['Alley'] = data['Alley'].fillna('None')            
data['MasVnrType'] = data['MasVnrType'].fillna('None')  
data['MasVnrArea'] = data['MasVnrArea'].fillna(data['MasVnrArea'].median()) 
data['BsmtQual'] = data['BsmtQual'].fillna(method = 'ffill')         
data['BsmtCond'] = data['BsmtCond'].fillna(method = 'ffill')         
data['BsmtExposure'] = data['BsmtExposure'].fillna(method = 'ffill')     
data['BsmtFinType1'] = data['BsmtFinType1'].fillna(method = 'ffill')     
data['BsmtFinType2'] = data['BsmtFinType2'].fillna(method = 'ffill')     
data['Electrical']=data['Electrical'].dropna()    
data['FireplaceQu'] = data['FireplaceQu'].fillna('None')
data['GarageType'] = data['GarageType'].fillna(method = 'ffill')
data['GarageYrBlt'] = data['GarageYrBlt'].fillna(method = 'ffill')     
data['GarageFinish'] = data['GarageFinish'].fillna(method = 'ffill')     
data['GarageQual'] = data['GarageQual'].fillna(method = 'ffill')       
data['GarageCond'] = data['GarageCond'].fillna(method = 'ffill')       
data['PoolQC'] = data['PoolQC'].fillna('None')           
data['Fence'] = data['Fence'].fillna('None')            
data['MiscFeature'] = data['MiscFeature'].fillna('None')   




datatest=pd.read_csv("C:/Users/PRADEEPHP/Music/HousePrice/test.csv")
null_columns_test = datatest.isnull().sum();
null_columns_test = null_columns_test[null_columns_test>0]

datatest['LotFrontage'] = datatest['LotFrontage'].fillna(datatest['LotFrontage'].mean())
datatest['Alley'] = datatest['Alley'].fillna('None')
datatest['Utilities'] = datatest['Utilities'].fillna(method = 'ffill')
datatest['Exterior1st'] = datatest['Exterior1st'].fillna(method = 'ffill') 
datatest['Exterior1st'] = datatest['Exterior1st'].fillna(method = 'ffill') 
datatest['Exterior2nd'] = datatest['Exterior2nd'].fillna('None')  
datatest['MasVnrArea'] = datatest['MasVnrArea'].fillna(data['MasVnrArea'].median()) 
datatest['MasVnrType'] = datatest['MasVnrType'].fillna(method = 'ffill') 
datatest['BsmtQual'] = datatest['BsmtQual'].fillna(method = 'ffill')         
datatest['BsmtCond'] = datatest['BsmtCond'].fillna(method = 'ffill')         
datatest['BsmtExposure'] = datatest['BsmtExposure'].fillna(method = 'ffill')     
datatest['BsmtFinType1'] = datatest['BsmtFinType1'].fillna(method = 'ffill')     
datatest['BsmtFinSF1'] = datatest['BsmtFinSF1'].fillna(method = 'ffill')     
datatest['BsmtFinType2'] = datatest['BsmtFinType2'].fillna(method = 'ffill')
datatest['BsmtFinSF2'] = datatest['BsmtFinSF2'].fillna(method = 'ffill')     
datatest['BsmtUnfSF'] = datatest['BsmtUnfSF'].fillna(method = 'ffill')     
datatest['TotalBsmtSF'] = datatest['TotalBsmtSF'].fillna(method = 'ffill')     
datatest['BsmtFullBath'] = datatest['BsmtFullBath'].fillna(method = 'ffill')     
datatest['BsmtHalfBath'] = datatest['BsmtHalfBath'].fillna(method = 'ffill')     
datatest['KitchenQual'] = datatest['KitchenQual'].fillna(method = 'ffill')     
datatest['Functional'] = datatest['Functional'].fillna(method = 'ffill')     
datatest['FireplaceQu'] = datatest['FireplaceQu'].fillna('None')     
datatest['GarageType'] = datatest['GarageType'].fillna(method = 'ffill')
datatest['GarageYrBlt'] = datatest['GarageYrBlt'].fillna(method = 'ffill')     
datatest['GarageFinish'] = datatest['GarageFinish'].fillna(method = 'ffill')     
datatest['GarageCars'] = datatest['GarageCars'].fillna(method = 'ffill')     
datatest['GarageArea'] = datatest['GarageFinish'].fillna(method = 'ffill')     
datatest['GarageQual'] = datatest['GarageArea'].fillna(method = 'ffill')       
datatest['GarageCond'] = datatest['GarageCond'].fillna(method = 'ffill')       
datatest['PoolQC'] = datatest['PoolQC'].fillna('None')           
datatest['Fence'] = datatest['Fence'].fillna('None')            
datatest['MiscFeature'] = datatest['MiscFeature'].fillna('None')   
datatest['SaleType'] = datatest['SaleType'].fillna('None')


train_encode = pd.get_dummies(datatest)
test_encode = pd.get_dummies(data.drop("SalePrice", axis=1))
test_encode_for_model = test_encode.reindex(fill_value =0,columns = train_encode.columns)

model = LinearRegression()
mod = model.fit(test_encode_for_model,y)

testresult=model.predict(train_encode)
testresult = pd.Series(testresult)
Id=datatest['Id']


testresult = pd.DataFrame({'Id': Id , 'SalePrice':testresult})
testresult.to_csv("C:/Users/PRADEEPHP/Music/HousePrice/predict.csv", index=False)
print(testresult)


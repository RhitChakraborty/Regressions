#Importing library
import numpy as np
import pandas as pd


#Importing dataset
dataset=pd.read_csv('50_Startups.csv')
#print(dataset)
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

#categorical encoding
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[3])],remainder='passthrough')
x=np.array(ct.fit_transform(x))
#print(x)

#avoiding the dummy variable trap
x=x[:,1:]


#Split in training set and test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#Multiple linear regression
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

#Prediction
pred=regressor.predict(x_test)
print(pred)

#backward elimination

# import statsmodels.api as sm
# x=np.append(arr=np.ones((50,1)).astype(int),values=x,axis=1)
# x_opt=np.array(x[:,[0,3,5]],dtype='float')
# regressor_ols=sm.OLS(endog=y,exog=x_opt).fit()
# print(regressor_ols.summary())
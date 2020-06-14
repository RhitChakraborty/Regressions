#Importing library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing dataset
dataset=pd.read_csv('Position_Salaries.csv')
#print(dataset)
x=dataset.iloc[:,1:-1].values
y=dataset.iloc[:,-1].values
# print(x)
# print(y)

# plt.scatter(x,y,color='red')
# plt.show()  #shows non linear pattern
# # # NO Split in training set and test set. since data is very less

#Linear regression
from sklearn.linear_model import LinearRegression
regressor_l=LinearRegression()
regressor_l.fit(x,y)

#Polynomial regression
from sklearn.preprocessing import PolynomialFeatures
regressor_poly=PolynomialFeatures(degree=4)
x_poly=regressor_poly.fit_transform(x)
#print(x_poly)
lin_reg2=LinearRegression()
lin_reg2.fit(x_poly,y)

##plot Linear Regrssion
# plt.scatter(x,y,color='red')
# plt.plot(x,regressor_l.predict(x),color='blue')
# plt.title('Linear Model')
# plt.xlabel('Position Level')
# plt.ylabel('Salary')
# plt.show()

#plot Polynomial Regression
x_grid=np.arange(min(x),max(x),0.1)     #array
x_grid=x_grid.reshape(len(x_grid),1)  #converting to matrix
plt.scatter(x,y,color='red')
#plt.plot(x,lin_reg2.predict(x_poly),color='blue')
plt.plot(x_grid,lin_reg2.predict(regressor_poly.fit_transform(x_grid)),color='blue')
plt.title('Polynomial Model')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

## Predict for Linear Regression
print(regressor_l.predict([[6.5]]))
## Predict for Polynomial Regression
print(lin_reg2.predict(regressor_poly.fit_transform([[6.5]])))
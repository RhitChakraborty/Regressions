#Importing library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing dataset
dataset=pd.read_csv('Salary_Data.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values


#Split in training set and test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

# print(x_train,x_test)
# print(y_train,y_test)

## Simple linear regression #####
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

#Predicting
pred=regressor.predict(x_test)
#print(pred,y_test)


#Visualizing the training set results
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('Salary Vs Workex,(Training set')
plt.xlabel('Years of WorkExperience')
plt.ylabel('Salary')
plt.show()

plt.scatter(x_test,y_test,color='red')
plt.plot(x_test,regressor.predict(x_test),color='blue')
plt.title('Salary Vs Workex (Testing set)')
plt.xlabel('Years of WorkExperience')
plt.ylabel('Salary')
plt.show()
testMSE=np.sqrt(np.square(np.subtract(pred,y_test)).mean())
print(testMSE)

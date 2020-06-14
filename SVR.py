#Importing library
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#Importing dataset
dataset=pd.read_csv('Position_Salaries.csv')
#print(dataset)
x=dataset.iloc[:,1:-1].values
y=dataset.iloc[:,-1].values

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
sc_y=StandardScaler()
x=sc_x.fit_transform(x)
y=sc_y.fit_transform(y.reshape(-1,1))

# SVR regression
from sklearn.svm import SVR
regressor=SVR(kernel='rbf')
regressor.fit(x,y)


#prediction
y_pred=sc_y.inverse_transform(regressor.predict(sc_x.transform(np.array([[6.5]]))))
print(y_pred)

#visualising SVR result
plt.scatter(x,y,color='red')
plt.plot(x,regressor.predict(x),color='blue')
plt.title('SVR Model')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

## more continuous graph
x_grid=np.arange(min(x),max(x),0.1)
x_grid=x_grid.reshape(-1,1)
plt.scatter(x,y,color='red')
plt.plot(x_grid,regressor.predict(x_grid),color='blue')
plt.title('SVR Model')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

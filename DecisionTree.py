import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#Importing dataset
dataset=pd.read_csv('Position_Salaries.csv')
#print(dataset)
x=dataset.iloc[:,1:-1].values
y=dataset.iloc[:,-1].values

# decision tree regression
from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=300,random_state=0)
regressor.fit(x,y)

#prediction
y_pred=regressor.predict([[6.5]])
print(y_pred)

#visualising SVR result
x_grid=np.arange(min(x),max(x),0.01)
x_grid=x_grid.reshape(-1,1)
plt.scatter(x,y,color='red')
plt.plot(x_grid,regressor.predict(x_grid),color='blue')
plt.title('Decision Tree Model')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

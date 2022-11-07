from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import median_absolute_error
dataset = pd.read_csv('Data.csv')
print(dataset.head(10))
X = dataset.iloc[:,:1] 
y = dataset.iloc[:, -1]

X
y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

X_test
y_test

X_train_new = np.array(X_train).reshape(-1, 1)
X_test_new = np.array(X_test).reshape(-1, 1)
y_train_new = np.array(y_train).reshape(-1, 1)
y_test_new = np.array(y_test).reshape(-1, 1)

regressor = LinearRegression()
regressor.fit(X_train_new, y_train_new)

regressor.score(X_train_new, y_train_new)
regressor.score(X_test_new, y_test_new)

# Predicting the Test set results
y_pred = regressor.predict(X_test_new)
y_pred 

y_pred_new=regressor.predict([[9.25]])
y_pred_new


mean_absolute_error(y_test_new, y_pred)


mean_squared_error(y_test_new, y_pred)


median_absolute_error(y_test_new, y_pred)

# Visualising the Training set results
plt.scatter(X_train_new, y_train_new, color = 'red')
plt.scatter(X_test_new, y_test_new, color = 'green')
plt.plot(X_train_new, regressor.predict(X_train_new), color = 'blue')
plt.title('Score for student')
plt.xlabel('Hours')
plt.ylabel('Score')
plt.show()




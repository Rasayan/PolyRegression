import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# print(x)
# print(y)

# from sklearn.model_selection import train_test_split
# x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2, random_state=0)
# print(x_train)
# print(x_test)
# print(y_train)
# print(y_test)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x, y)

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=5)
x_poly = poly.fit_transform(x)
lin_reg_2 = LinearRegression()

lin_reg_2.fit(x_poly, y)

print(x_poly)

plt.scatter(x, y, color = 'red')
plt.plot(x,lr.predict(x),color='blue')
plt.title('Truth or Bluff(LR)')
plt.xlabel('Position')
plt.ylabel('Salaries')
# plt.show()

plt.scatter(x, y, color = 'red')
plt.plot(x,lin_reg_2.predict(x_poly),color='blue')
plt.title('Truth or Bluff(LR)')
plt.xlabel('Position')
plt.ylabel('Salaries')
plt.show()

X_grid = np.arange(min(x), max(x), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(x, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
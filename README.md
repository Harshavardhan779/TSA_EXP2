# Ex.No: 02 LINEAR AND POLYNOMIAL TREND ESTIMATION
### Developed by: HARSHAVARDHAN
### Register No: 2122222400114
Date:
### AIM:
To Implement Linear and Polynomial Trend Estiamtion Using Python.

### ALGORITHM:
1. Import necessary libraries (NumPy, Matplotlib)
2. Load the dataset
3. Calculate the linear trend values using least square method
4. Calculate the polynomial trend values using least square method
5. End the program
### PROGRAM:
#### A - LINEAR TREND ESTIMATION
```python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
data=pd.read_csv('infy_stock-Copy1.csv')
data.head()
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date')
data['Date'] = data['Date'].apply(lambda x: x.toordinal())
X = data['Date'].values.reshape(-1, 1)
y = data['Volume'].values
linear_model = LinearRegression()
linear_model.fit(X, y)
data['Linear_Trend'] = linear_model.predict(X)
plt.figure(figsize=(10,6))
plt.plot(data['Date'], data['Volume'],label='Original Data')
plt.plot(data['Date'], data['Linear_Trend'], color='orange', label='Linear Trend')
plt.title('Linear Trend Estimation')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.legend()
plt.grid(True)
plt.show()
```
#### B- POLYNOMIAL TREND ESTIMATION
```python
poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(X)
poly_model = LinearRegression()
poly_model.fit(X_poly, y)
data[' Volume'] = poly_model.predict(X_poly)
plt.figure(figsize=(10,6))
plt.bar(data['Date'], data['Volume'], label='Original Data', alpha=0.6)
plt.plot(data['Date'], data[' Volume'],color='yellow', label='Poly Trend(Degree 2)')
plt.title('Polynomial Trend Estimation')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.legend()
plt.grid(True)
plt.show()
```
### Dataset:

### OUTPUT
A - LINEAR TREND ESTIMATION
![linear](https://github.com/user-attachments/assets/cdaf01b8-7d4e-4ca5-9efc-bd46e6ee8220)

B- POLYNOMIAL TREND ESTIMATION
![poly](https://github.com/user-attachments/assets/eac23dfb-3d59-499d-b000-d52897d4720c)


### RESULT:
Thus the python program for linear and Polynomial Trend Estiamtion has been executed successfully.

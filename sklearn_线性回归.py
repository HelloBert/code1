import numpy as np
from sklearn.linear_model import LinearRegression

X1 = np.random.rand(100, 1)
X2 = np.random.rand(100, 1)

x_b = np.c_[X1, X2]
y = 3 + 4*X1 + 5*X2 + np.random.randn(100, 1)

reg = LinearRegression()
reg.fit(x_b, y)
print(reg.intercept_, reg.coef_)

print(reg.predict(np.array([[3, 4]])))
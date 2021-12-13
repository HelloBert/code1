import numpy as np
import matplotlib.pyplot as plt

# 多元线性回归
np.random.seed(40)
X1 = np.random.rand(100, 1)
X2 = np.random.rand(100, 1)
x_b = np.c_[np.ones((100, 1)), X1, X2]

y = 3 + 4*X1 + 5*X2 + np.random.randn(100, 1)

theta = np.linalg.inv(x_b.T.dot(x_b)).dot(x_b.T).dot(y)
print(theta)

x_test = np.array([[1, 2]])
x_test_b = np.c_[np.ones((1, 1)), x_test]
y_predict = x_test_b.dot(theta)
print(y_predict)
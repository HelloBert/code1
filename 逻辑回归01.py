import math

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from mpl_toolkits.mplot3d import Axes3D


# 定义X，y
data = load_breast_cancer()
X, y = scale(data['data'][:, :2]), data['target']

Lr = LogisticRegression(fit_intercept=False)
Lr.fit(X, y)

# 训练得到theta1,theta2
theta1 = Lr.coef_[0, 0]
theta2 = Lr.coef_[0, 1]
print(theta1, theta2)

# 定义sigmoid函数
def sigmoid_fun(x, w1, w2):
    z = x[0] * w1 + x[1] * w2
    return 1 / (1 + np.exp(-z))

def loss_fun(sample_features, sample_labels, W1, W2):
    result = 0
    for feature, label in zip(sample_features, sample_labels):
        predict = sigmoid_fun(feature, W1, W2)
        loss = -1 * label*np.log(predict) - (1-label) * np.log(1-predict)
        result += loss
    return result

loss_result = loss_fun(X, y, theta1, theta2)



W1_linspace = np.linspace(theta1-0.6, theta1+0.6, 50)
#print(W1_linspace)
W2_linspace = np.linspace(theta2-0.6, theta2+0.6, 50)
#
#
loss_result1 = np.array([loss_fun(X, y, i, theta2) for i in W1_linspace])
loss_result2 = np.array([loss_fun(X, y, theta1, i) for i in W2_linspace])
#
fig1 = plt.figure(figsize=(8, 6))
#
plt.subplot(2, 2, 1)
plt.plot(W1_linspace, loss_result1)

plt.subplot(2, 2, 2)
plt.plot(W2_linspace, loss_result2)

plt.subplot(2, 2, 3)
theta1_grid, theta2_grid = np.meshgrid(W1_linspace, W2_linspace)
loss_result3 = loss_fun(X, y, theta1_grid, theta2_grid)
plt.contour(theta1_grid, theta2_grid, loss_result3, 30)

fig2 = plt.figure()
ax = Axes3D(fig2)
ax.plot_surface(theta1_grid, theta2_grid, loss_result3)


plt.show()








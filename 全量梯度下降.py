import numpy as np


#创建数据
X = np.random.rand(100, 1)
x_b = np.c_[np.ones((100, 1)), X]

y = 3 + 4*X + np.random.randn(100, 1)

# 设置超参数
learning_rate = 0.0001
n_interations = 10000

theta = np.random.rand(2, 1)

# 迭代进行梯度下降
for _ in range(n_interations):
    g = x_b.T.dot(x_b.dot(theta)-y)
    theta = theta - learning_rate * g

print(theta)
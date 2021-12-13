import numpy as np

X = np.random.rand(100, 1)
x_b = np.c_[np.ones((100, 1)), X]

y = 5 + 6*X + np.random.randn(100, 1)

# 设置超参数
learning_rate = 0.001
epoches = 5000000
m = 100

theta = np.random.randn(2, 1)
for _ in range(epoches):
    #for _ in range(m):
        random_index = np.random.randint(m)
        x_train = x_b[random_index:random_index+1]
        y_train = y[random_index:random_index+1]
        g = x_train.T.dot(x_train.dot(theta)-y_train)
        theta = theta - learning_rate * g

print(theta)
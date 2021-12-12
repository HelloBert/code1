import numpy as np
import matplotlib.pyplot as plt


# 手动创建数据
# x是一百行一列
# 测试修改
X = np.random.rand(100, 1)

# y是100行1列，这里是真实的数据y_hat + error
y = 4 + 5*X + np.random.randn(100, 1)

x_b = np.c_[np.ones((100, 1)), X]

theta = np.linalg.inv(x_b.T.dot(x_b)).dot(x_b.T).dot(y)
print(theta)


# 创建测试集x
x_test = np.array([[0],
                   [1]])
x_test_b = np.c_[np.ones((2, 1)), x_test]

y_predict = x_test_b.dot(theta)
print(y_predict)

plt.plot(x_test, y_predict, 'r-')
plt.plot(X, y, 'b.')
plt.axis([0, 1, 0, 12])
plt.show()
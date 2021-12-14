import numpy as np

X = np.random.rand(100, 1)
x_b = np.c_[np.ones((100, 1)), X]

y = 4 + 5*X + np.random.randn(100, 1)


learning_rate = 0.001
epoches = 10000
m = 100
batch_size = 10
num_batches = int(m/batch_size)

theta = np.random.randn(2, 1)

for _ in range(epoches):
    arr = np.arange(len(x_b))
    np.random.shuffle(arr)
    x_b = x_b[arr]
    y = y[arr]
    for arr_index in range(m):
        #random_index = np.random.randint(m)
        x_batch = x_b[arr_index:arr_index+10]
        y_batch = y[arr_index:arr_index+10]
        g = x_batch.T.dot(x_batch.dot(theta)-y_batch)
        theta = theta - learning_rate*g

print(theta)
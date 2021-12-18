from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import numpy as np

data = load_iris()
# 这里如果是2那么返回True，如果返回其他那么返回False
y = (data['target'] == 2).astype(np.int)


x = data['data'][:, 3:]

multi_classifier = LogisticRegression(solver='sag', max_iter=1000)
multi_classifier.fit(x, y)

x_new = np.linspace(1, 3, 1000).reshape(-1, 1)
y_predict = multi_classifier.predict(x_new)
print(y_predict)



import pandas as pd
import numpy as np
from sklearn.datasets import load_iris                  # 导入数据集
from sklearn.tree import DecisionTreeClassifier         # 分类树
from sklearn.tree import export_graphviz                # 画图
from sklearn.model_selection import train_test_split    # 切分训练集和测试集
from sklearn.metrics import accuracy_score              # 计算准确率
import matplotlib.pyplot as plt
import matplotlib as mpl

iris = load_iris()
data = pd.DataFrame(iris.data)
data.columns = iris.feature_names
data['species'] = iris.target
print(data.head(5))

x = data.iloc[:, 2:4]           # petal length (cm)  petal width (cm)  花瓣的长度和宽度
y = data.iloc[:, -1]

# 切分训练集和测试集X,y
X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=43)

tree_clf = DecisionTreeClassifier(max_depth=8, criterion='gini')
tree_clf.fit(X_train, y_train)

y_test_hat = tree_clf.predict(X_test)
print('acc score',accuracy_score(y_test_hat, y_test))

# 打印两个维度的重要性
print(tree_clf.feature_importances_)
# [0.0824092 0.9175908]


depth = np.arange(1, 15)
err_list = []
for d in depth:
    clf = DecisionTreeClassifier(max_depth=d, criterion='gini')
    clf.fit(X_train, y_train)
    y_test_hat = clf.predict(X_test)
    print(d, '错误率：%.2f%%'%(accuracy_score(y_test_hat, y_test)))


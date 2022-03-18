from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets import load_boston


# 鸢尾花数据集做分类
li = load_iris()
X = li.data
y = li.target
# 获取特征值
print(li.data)
# 获取目标值
print(li.target)
# 打印数据集描述
print(li.DESCR)

# 训练集占75%，测试集占25%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# 获取新闻数据集
news = fetch_20newsgroups(subset='all')
print(news.data)
print(news.target)

# 波士顿房价数据集
bost = load_boston()
print(bost.data)
print(bost.target)
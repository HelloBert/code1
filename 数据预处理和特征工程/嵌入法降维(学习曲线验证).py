import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
import numpy as np
from sklearn.model_selection import cross_val_score

lbc_data = load_breast_cancer()
X = lbc_data.data
y = lbc_data.target

rfc = RandomForestClassifier(n_estimators=100, random_state=10)
"""
# data = SelectFromModel(rfc, threshold=0.02).fit_transform(lbc_data.data, lbc_data.target)
# score = cross_val_score(rfc, data, y, cv=5).mean()
# print(score)

# threshold是根据每个值的重要程度进行选择，这个threshold如何取值?
# 画出学习曲线

max_feature = (rfc.fit(X, y).feature_importances_).max()
line_data = np.linspace(0, max_feature, 20)
print(line_data)
"""

"""
scores = []

for i in line_data:
    data = SelectFromModel(rfc, threshold=i).fit_transform(X, y)
    score = cross_val_score(rfc, data, y, cv=5).mean()
    scores.append(score)

plt.figure(figsize=(15, 5))
plt.plot(line_data, scores, color='black', label='学习曲线')
plt.show()
"""

# 根据学习曲线返现最高值在0.01到0.027之间，那么我们在这个数段之间选取20个数，以帮助我们寻找最大值
"""
thirdhold = np.linspace(0.01, 0.027, 20)

score_ = []
for i in thirdhold:
    data = SelectFromModel(rfc, threshold=i).fit_transform(X, y)
    score = cross_val_score(rfc, data, y, cv=5).mean()
    score_.append(score)
print(max(score_))
plt.figure(figsize=(15, 5))
plt.plot(thirdhold, score_)
plt.show()
"""

# 经过学习曲线进一步验证，我们发现分数最高值是0.9578326346840551，在图像上显示大概在0.016附近，那么我们就拿0.016做thirdhold参数进行训练
data = SelectFromModel(rfc, threshold=0.016).fit_transform(X, y)
score = cross_val_score(rfc, data, y, cv=5).mean()
print(score)
# 0.9578326346840551结果正好是最好的结果


# 我们已经将threshold调到最好了，那么我们可以在试着调试一下随机森林n_estimators，我们发现n_estimators从10调到100后，准去率直接达到0.9631113181183046









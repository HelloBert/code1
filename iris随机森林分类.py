from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

iris = load_iris()

X = iris.data[:, :]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45)
# 创建模型，n_estimators=15，15颗小树，max_leaf_nodes=16，最多16个叶子节点，n_jobs=1并行度
rnd_clf = RandomForestClassifier(n_estimators=15, max_leaf_nodes=16, n_jobs=1, oob_score=True)
rnd_clf.fit(X_train, y_train)
y_pred_rf = rnd_clf.predict(X_test)
print(rnd_clf.oob_score_)
print(accuracy_score(y_test, y_pred_rf))

# 打印每个特征的重要度
for name, score in zip(iris.feature_names, rnd_clf.feature_importances_):
    print(name, score)
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def iris_Knn():

    # 加载数据集
    iris = load_iris()
    data = pd.DataFrame(iris.data)
    X = data.iloc[:, 2:4]
    y = iris.target

    # 切分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # 分类
    knn_iris = KNeighborsClassifier(n_neighbors=5)
    knn_iris.fit(X_train, y_train)
    y_predict = knn_iris.predict(X_test)
    print("预测目标位置为", y_predict)
    print("预测准确率为", knn_iris.score(X_test, y_test))












if __name__ == "__main__":
    iris_Knn()
import pandas as pd
from sklearn.feature_extraction import DictVectorizer    # 字典特征抽取
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

def descision():
    """
    决策树对泰坦尼克号上面的人员预测生死
    :return:
    """

    # 获取数据
    taitan = pd.read_csv('C:/Users/10509/Desktop/AI练习数据/titanic.csv')

    # 选择特征和目标值
    x = taitan[['pclass', 'age', 'sex']]
    y = taitan['survived']

    # 缺失值处理
    x['age'].fillna(x['age'].mean(), inplace=True)

    # 分割训练集测试集
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    # 特征工程，当特征里面是类别的时候，需要做OneHot编码
    dict = DictVectorizer(sparse=False)
    X_train = dict.fit_transform(X_train.to_dict(orient="records"))
    X_test = dict.transform(X_test.to_dict(orient="records"))
    print(X_train)

    # 用决策树进行预测
    dec = DecisionTreeClassifier(max_depth=8)
    dec.fit(X_train, y_train)
    y_predict = dec.predict(X_test)
    print("准确率是:", dec.score(X_test, y_test))




if __name__ == "__main__":
    descision()
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


def pre_data():
    boston = load_boston()
    X = boston.data
    y = boston.target

    # 确定一个随机种子
    rng = np.random.RandomState(10)

    # 给完整的数据添加一些缺失值，为了测试Imputer
    n_samples = X.shape[0]
    n_feature = X.shape[1]

    missing_rate = 0.5
    # np.floor()向下取整返回一个.0的浮点数，再用int取整数。
    n_miss_samples = int(np.floor(n_samples * n_feature * missing_rate))

    # 取所有随机的横坐标纵坐标
    missing_features = rng.randint(0, n_feature, n_miss_samples)
    missing_samples = rng.randint(0, n_samples, n_miss_samples)

    X_missing = X.copy()    # X_missing数据进行处理
    y_missing = y.copy()    # y_missing不进行处理

    X_missing[missing_samples, missing_features] = np.nan


    # 用sklearn自带的方法填充空值
    # sp = SimpleImputer(missing_values=np.nan, strategy="mean")
    # X = sp.fit_transform(X_missing)
    # print(pd.DataFrame(X).info())

    # 用随机森林做预测进行填充空值
    X_missing_reg = X_missing.copy()

    for i in range(13):
        X_missing_reg = pd.DataFrame(X_missing_reg)
        sortindex = np.argsort(X_missing_reg.isnull().sum(axis=0)).values

        # 将第四列变成目标值
        fillc = X_missing_reg.iloc[:, i]

        df = X_missing_reg
        # 将去掉第i列剩下的列和Y合并成一个df
        df = pd.concat([df.iloc[:, df.columns != i], pd.DataFrame(y_missing)], axis=1)

        # 填充0
        df_0 = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=0).fit_transform(df)

        Ytrain = fillc[fillc.notnull()]
        Ytest = fillc[fillc.isnull()]

        Xtest = df_0[Ytest.index, :]
        Xtrain = df_0[Ytrain.index, :]

        # 随机森林训练模型
        rfc = RandomForestRegressor(n_estimators=100)
        rfc.fit(Xtrain, Ytrain)
        # 随机森林预测出来剩下的标签值
        Ypredict = rfc.predict(Xtest)

        #print(X_missing_reg.iloc[:, 4].isnull())

        # 用loc传入内容返回对应的索引
        X_missing_reg.loc[X_missing_reg.iloc[:, i].isnull(), i] = Ypredict

    # # 查看有无缺失值
    # print(X_missing_reg.isnull().sum())
    return X_missing_reg, y


def Reg(data, target):
    # 建立模型
    Rfr = RandomForestRegressor(n_estimators=200,
                                criterion='mse',
                                max_depth=5)
    # 交叉验证
    print(cross_val_score(Rfr, data, target, scoring="neg_mean_squared_error", cv=10).mean())



if __name__ == "__main__":
    X, y = pre_data()
    Reg(X, y)
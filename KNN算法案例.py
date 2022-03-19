from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV



def knncls():
    """
    K-近邻预测用户签到位置
    :return:
    """
    # 读取数据
    data = pd.read_csv("C:/Users/10509/Desktop/AIStudy/FBlocation/train.csv")
    # 处理数据
    # 1.缩小数据，通过pandas查询数据筛选
    data = data.query("x>1.0 & x<1.25 & y>2.5 & y<2.75")

    # 2.处理时间
    time_value = pd.to_datetime(data['time'], unit='s')
    # 把日期格式转换成字典格式
    time_value = pd.DatetimeIndex(time_value)

    # 3.构造一些特征
    data['day'] = time_value.day
    data['hour'] = time_value.hour
    data['weekday'] = time_value.weekday

    # 4.删除原来的时间戳
    data = data.drop(['time'], axis=1)

    # 5.把签到数量少于n个目标位置删除
    place_count = data.groupby('place_id').count()
    tf = place_count[place_count.row_id>3].reset_index()
    data = data[data['place_id'].isin(tf.place_id)]

    # 6.取出数据中的特征值和目标值
    y = data['place_id']
    # data = data.drop(['place_id'], axis=1)
    x = data.drop(['row_id'], axis=1)
    print(x)

    # 7.把数据分成训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    # 特征工程（标准化）对训练集和测试集的特征值进行归一化
    std = StandardScaler()
    X_train = std.fit_transform(X_train)
    X_test = std.fit_transform(X_test)


    # 训练数据
    knn = KNeighborsClassifier(n_neighbors=5)
    #
    # # fit, predice, score
    # knn.fit(X_train, y_train)
    # y_predict = knn.predict(X_test)
    # print("预测目标位置为", y_predict)
    #
    # # 得出准确率
    # print("预测准确率", knn.score(X_test, y_test))

    param = {"n_neighbors": [3, 5, 10]}

    # 进行网格搜索
    gc = GridSearchCV(knn, param_grid=param, cv=2)
    gc.fit(X_train,y_train)
    gc.predict(X_test)
    print("测试集上的准确率:", gc.score(X_test, y_test))
    print("在交叉验证中最好的结果：", gc.best_score_)
    print("选择的最好的模型是:", gc.best_estimator_)
    print("每个超参数交叉验证的结果：", gc.cv_results_)


    return None




if __name__ == "__main__":
    knncls()
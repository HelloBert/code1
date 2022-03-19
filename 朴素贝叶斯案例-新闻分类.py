from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report




def naviebayes():

    """
    朴素贝叶斯进行文本分类
    :return:
    """
    news = fetch_20newsgroups(subset='all')
    X = news.data
    y = news.target
    print(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # 对数据进行特征抽取
    news_tf = TfidfVectorizer()
    # 以训练集当中的词的列表进行每篇文章重要性统计
    X_train = news_tf.fit_transform(X_train)
    # print(news_tf.get_feature_names())
    X_test = news_tf.transform(X_test)

    # 进行朴素贝叶斯算法的预测
    mlt = MultinomialNB(alpha=1.0)
    mlt.fit(X_train, y_train)
    y_predict = mlt.predict(X_test)

    print("预测得到的分类是：", y_predict)
    print("准确率是：",mlt.score(X_test, y_test))
    print("每个类别的精准率和召回率：", classification_report(y_test, y_predict, target_names=news.target_names))


    return None


if __name__ == "__main__":
    naviebayes()


# [10  3 17 ...  3  1  7]
# 预测得到的分类是： [11  2 10 ...  8  5  6]
# 准确率是： 0.8406196943972836
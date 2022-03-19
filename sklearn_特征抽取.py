import jieba
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA

def pca():
    # 主成分分析进行降维
    # 信息的保留量90%-100%之间
    pca = PCA(n_components=0.95)
    data = pca.fit_transform([[80, 79, 1, 14], [80, 111, 33, 754], [80, 234, 11, 23]])
    print(data)


# sklearn特征选择,删除低方差的特征
def var():
    va = VarianceThreshold(threshold=0)
    data = va.fit_transform([[80, 79, 1, 14], [80, 111, 33, 754], [80, 234, 11, 23]])
    print(data)



# sklearn对缺失值处理
def im():
    # strategy='mean'按平均值填补，按照列计算
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    data = imputer.fit_transform([[89, 79, np.nan, 14], [78, 111, 33, np.nan], [82, 234, np.nan, 23]])
    print(data)

# 标准归一化
def standscaler():
    ss = StandardScaler()
    data = ss.fit_transform([[89, 79, 1, 14], [78, 111, 33, 754], [82, 234, 11, 23]])
    print(data)

# 数据归一化
def Maxminscale():
    # feature_range指定2-3之间的范围
    mm = MinMaxScaler(feature_range=(2, 3))
    data = mm.fit_transform([[89, 79, 1, 14], [78, 111, 33, 754], [82, 234, 11, 23]])
    print(data)


# 字典特征抽取
def dictvec():
    # 特征数据是字符串的话不能输入到算法里面，是要进行特征，转换的转换成OneHot编码。有利于机器学习算法分析。
    # 字典数据抽取,把字典中的一些类别特征，转换乘特征（数字），但是字典里面的数字不会进行转换，因为本来就是数据。
    # 如果是数组形式，有类别的这些特征，我们要先转换成字典，再进行数据抽取。
    # 实例化
    dict = DictVectorizer()

    # 返回的data是一个sparse矩阵格式
    # sparse节约内存，方便数据处理
    data = dict.fit_transform([{"city": "北京", 'temperature': 100}, {"city": "上海", 'temperature': 60}, {"city": "深圳", 'temperature': 30}])

    print(dict.inverse_transform(data))
    print(data)
    # 返回内容列表
    print(dict.get_feature_names())

# 文本特征抽取
def countvec():

    test = CountVectorizer()        # 统计次数
    data = test.fit_transform(["life is is short I like python", "life is too long,I dislike python"])
    # 词去重，放在一个列表中
    print(test.get_feature_names())
    # 返回的是sparse格式，toarray手动转成二元组形式，对每篇文章在词的列表里面统计每个词出现的次数（单个字母不统计）
    print(data.toarray())

    # ['dislike', 'is', 'life', 'like', 'long', 'python', 'short', 'too']
    # [[0 1 1 1 0 1 1 0]
    #  [1 1 1 0 1 1 0 1]]

# 中文特征抽取
# 中文因为没有像英文一样进行分词，所以在做特征抽取前要先用jieba进行分词
def cutword():
    # 用jieba分词
    cont1 = jieba.cut("今天很残酷，明天很残酷，后天很美好")
    cont2 = jieba.cut("我们看到的从很远星系来的光是几百万年前发出的")
    cont3 = jieba.cut("如果只用一种方式了解某样事物，你就不会真正了解他")

    # 转换成列表
    content1 = list(cont1)
    content2 = list(cont2)
    content3 = list(cont3)
    # print(content3)
    # ['如果', '只用', '一种', '方式', '了解', '某样', '事物', '，', '你', '就', '不会', '真正', '了解', '他']
    # 把列表转换成字符串
    c1 = ' '.join(content1)
    c2 = ' '.join(content2)
    c3 = ' '.join(content3)
    # print(c1)
    # 今天 很 残酷 ， 明天 很 残酷 ， 后天 很 美好

    return c1, c2, c3


def hanzivec():
    test = CountVectorizer()
    c1, c2 ,c3 = cutword()

    data = test.fit_transform([c1, c2, c3])
    print(test.get_feature_names())
    print(data.toarray())

# TFIDF特征抽取
def tfidfvec():
    test = TfidfVectorizer()
    c1, c2, c3 = cutword()
    data = test.fit_transform([c1, c2, c3])
    print(test.get_feature_names())
    print(data.toarray())





if __name__ == "__main__":
    #dictvec()
    print("------------------------")
    #countvec()
    #hanzivec()
    #tfidfvec()
    #Maxminscale()
    #standscaler()
    # im()
    # var()
    pca()
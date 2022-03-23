import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import  PCA

import numpy as np

def pca_test():
    # 导入数据集
    data = load_iris()
    X = data.data
    y = data.target

    """
    # 对所有特征数据进行PCA降维, mle让sklearn自己选择最好的取值（但是计算量太大，如果数据量态度，有可能崩掉）
    pca = PCA(n_components='mle')
    X_pca = pca.fit_transform(X)
    """

    # 直接设置保留多少百分比的信息量
    pca_f = PCA(n_components=0.97, svd_solver='full')
    pca_f.fit_transform(X)



    # 查看降维后每个新特征上所带的信息量大小,大部分有效特征都会集中在第一个特征上
    print(pca_f.explained_variance_)
    # [4.22824171 0.24267075]

    # 查看降维后新特征向量信息量所占原始数据信息量百分比(可解释性方差比)
    print(pca_f.explained_variance_ratio_.sum())
    # [0.92461872 0.05306648]这两个数据加和就是原有数据保留的信息占比


    """
    # 画图展示如果n_components什么都不输入，随着n_components维度的增加，信息所占比变化情况，然后根据信息所占比与维度的个数选择合适的维度
    plt.figure()
    plt.plot([1, 2, 3, 4], np.cumsum(pca.explained_variance_ratio_))
    plt.xticks([1, 2, 3, 4])
    #plt.legend()
    plt.show()
    """




    """
    # 画图展示
    plt.figure()
    # scatter是专门用来画散点图的
    plt.scatter(X_pca[y==0, 0], X_pca[y==0, 1], c='red', label=data.target_names[0])
    plt.scatter(X_pca[y==1, 0], X_pca[y==1, 1], c='black', label=data.target_names[1])
    plt.scatter(X_pca[y==2, 0], X_pca[y==2, 1], c='orange', label=data.target_names[2])
    # 画图例
    plt.legend()
    plt.title("PCA of iris datasets")
    plt.show()
    """

    """
    # 用循环的方式画出图
    color = ['red', 'black', 'orange']
    plt.figure()
    for i in [0, 1, 2]:
        plt.scatter(
            X_pca[y==i, 0],
            X_pca[y==i, 1],
            c=color[i],
            # alpha透明度
            alpha=.7,
            label=data.target_names[i],
        )
    plt.legend()
    plt.title("PCA of ifis datases")
    plt.show()
    """






if __name__ == "__main__":
    pca_test()

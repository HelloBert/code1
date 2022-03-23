from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


def dgs():
    digits = load_digits()
    X = digits.data
    print(X.shape)
    y = digits.target
    print(y)

    rng = np.random.RandomState(42)
    # 从原数据集中抽取符合正态分布，方差是2的数据集,是一个带噪音的数据集
    noisy = rng.normal(X, 2)

    pca = PCA(32)
    # 对带噪声的数据降维
    x_dr = pca.fit_transform(noisy)
    """
    # 还原
    x_dr_inverse = pca.inverse_transform(x_dr)
    print(x_dr_inverse.shape)

    fig, axes = plt.subplots(2, 10,
                             figsize=(8, 4),
                             subplot_kw={"xticks":[], "yticks":[]})

    for i in range(10):
        axes[0, i].imshow(noisy[i, :].reshape(8, 8), cmap="gray")
        axes[1, i].imshow(x_dr_inverse[i, :].reshape(8, 8), cmap="gray")
    """


    scores = []

    for i in range(10, 60, 10):
        x_dr = PCA(i).fit_transform(X)
        RFC = RandomForestClassifier(n_estimators=100, random_state=30)
        once = cross_val_score(RFC, x_dr, y, cv=5).mean()
        scores.append(once)
    plt.figure(figsize=(10, 5))
    plt.plot(range(10, 60, 10), scores)
    plt.show()




if __name__ == "__main__":
    dgs()
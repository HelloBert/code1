from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

face = fetch_lfw_people(min_faces_per_person=60)
print(face.images.shape)
# (1348, 62, 47)
# 1348是矩阵中图像的个数，62是每个图象特征矩阵的行，47是每个特征矩阵的列，说白了共有1348张图，每张图62行47列
print(face.data.shape)

X = face.data
y = face.target


pca = PCA(150)
X_dr = pca.fit_transform(X)
print(X_dr.shape)


X_inverse = pca.inverse_transform(X_dr)
print(X_inverse.shape)

"""
fig, axes = plt.subplots(4, 5,
              figsize=(8, 4),
              subplot_kw={"xticks":[], "yticks":[]})

for i, ax in enumerate(axes.flat):
    ax.imshow(face.images[i, :, :], cmap="gray")

fig, axes = plt.subplots(4, 5,
              figsize=(8, 4),
              subplot_kw={"xticks":[], "yticks":[]})

for i, ax in enumerate(axes.flat):
    ax.imshow(X_inverse[i, :].reshape(62, 47), cmap="gray")

# plt.show()
"""

fig, axes = plt.subplots(2, 10,
                         subplot_kw={"xticks":[], "yticks":[]},
                         figsize=(8, 4))
for i in range(10):
    axes[0, i].imshow(face.images[i, :, :], cmap="gray")
    axes[1, i].imshow(X_inverse[i,:].reshape(62, 47), cmap="gray")

plt.show()





"""
# 创建画布
fig, axes = plt.subplots(4, 5,
                         figsize=(8, 4),
                         subplot_kw={"xticks":[], "yticks":[]},
                         )
# 将图片填充画布
for i, ax in enumerate(axes.flat):
    ax.imshow(face.images[i, :, :], cmap="gray")

pca = PCA(150).fit(X)
V = pca.components_
print(V.shape)

fig, axes = plt.subplots(3, 8,
             subplot_kw={"xticks":[], "yticks":[]},
             figsize=(8, 4))
for i, ax in enumerate(axes.flat):
    ax.imshow(V[i, :].reshape(62, 47),cmap="gray")
plt.show()
"""


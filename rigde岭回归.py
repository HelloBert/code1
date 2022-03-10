import numpy as np
from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Lasso


X = 2*np.random.rand(100,1)
y = 4 + 3*X + np.random.randn(100, 1)


# 使用梯度下降的方法， penalty代表使用哪种正则项，max_iter代表迭代多少次

lasso_rag = Lasso(alpha=0.15, max_iter=30000)
lasso_rag.fit(X, y)
print(lasso_rag.predict([[1.5]]))
print(lasso_rag.intercept_)
print(lasso_rag.coef_)
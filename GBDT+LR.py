from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score

class GradientBoostingWithLR(object):
    def __init__(self):
        self.gbdt_model = None
        self.lr_model = None
        self.gbdt_encoder = None
        self.X_train_leafs = None
        self.X_test_leafs = None
        self.X_trans = None

    # gbdt训练一个模型
    def gbdt_train(self, x, y):
        # n_estimators弱学习器的数量，max_depth最大深度，max_features所有特征的百分之五十
        gbdt_model = GradientBoostingClassifier(n_estimators=5, max_depth=2, max_features=0.5)
        gbdt_model.fit(x, y)
        return gbdt_model

    # lr训练一个模型
    def lr_train(self, x, y):
        lr_model = LogisticRegression()
        lr_model.fit(x, y)
        return lr_model

    # 传入数据得到gbdt模型， 这个模型对x数据进行处理做OneHot编码，转换后在做lr训练，得到最终模型

    def gbdt_lr_train(self, x, y):
        self.gbdt_model = self.gbdt_train(x, y)
        # apply会把x训练集每条样本落在哪个叶子节点返回
        self.X_train_leafs = self.gbdt_model.apply(x)[:, :, 0]

        # 做Onehot编码
        self.gbdt_encoder = OneHotEncoder()
        self.X_trans = self.gbdt_encoder.fit_transform(self.X_train_leafs)

        # 把转换之后的数据进行lr训练
        self.lr_model = self.lr_train(self.X_trans, y)
        return self.lr_model

    # 用训练数据得到上面的模型，在传入测试及数据x_test, y_test，做OneHot编码，再用上面的训练模型做预测
    def gbdt_lr_pred(self, model, x, y):
        self.X_test_leafs = self.gbdt_model.apply(x)[:, :, 0]
        test_rows, cols = self.X_test_leafs.shape
        print(test_rows, cols)
        x_trans = self.gbdt_encoder.fit_transform(self.X_test_leafs)
        y_pred = model.predict_proba(x_trans)[:, 1]
        auc_score = roc_auc_score(y, y_pred)
        print("GBDT + LR AUC score: %.5F" % auc_score)
        return auc_score

    # 直接用gbdt做分类
    def model_assessment(self, model, x, y, model_name='GBDT'):
        y_pred = model.predict_proba(x)[:, 1]
        auc_score = roc_auc_score(y, y_pred)
        print('%s AUC score: %.5f' %(model_name, auc_score))
        return auc_score

# 加载数据
def load_data():
    iris_data = load_iris()
    x = iris_data['data']
    y = iris_data['target'] == 2
    return train_test_split(x, y, test_size=0.4, random_state=0)


if __name__ == "__main__":
    x_train, x_test, y_train, y_test = load_data()
    gblr = GradientBoostingWithLR()
    # 训练模型
    gbdt_lr_model = gblr.gbdt_lr_train(x_train, y_train)
    gblr.model_assessment(gblr.gbdt_model, x_test, y_test)
    # 做预测
    gblr.gbdt_lr_pred(gbdt_lr_model, x_test, y_test)




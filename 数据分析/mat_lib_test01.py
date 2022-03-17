import random
from sklearn.feature_extraction import DictVectorizer
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import font_manager

x = [i for i in range(1, 121)]
y = [random.randint(20,35) for i in range(120)]

# 设置字体
my_font = font_manager.FontProperties(fname='./PingFang.ttc')

plt.figure(figsize=(20, 8), dpi=80)
# 调整x轴刻度
_xtick_label = ["10点{}分".format(i) for i in range(60)]
_xtick_label += ["11点{}分".format(i) for i in range(60)]


# 传入字符串的时候，要把数据也传进去，取步长，数据长度一样一一对应
# totation=90,旋转90度
plt.xticks(x[::3], _xtick_label[::3], rotation=45)

plt.plot(x, y)
plt.show()


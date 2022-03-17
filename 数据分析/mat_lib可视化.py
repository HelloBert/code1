from matplotlib import pyplot as plt


# 准备x轴的值
x = range(2, 26, 2)
# 准备y轴的值
y = [12, 15, 13, 33, 44, 22, 21, 21, 23, 10, 12, 13]

# 设置图片大小
plt.figure(figsize=(20, 8), dpi=80)

# 设置x轴刻度
_xtick_labels = [ i/2 for i in range(4, 49)]
plt.xticks(_xtick_labels[::3])
plt.yticks(range(min(y),max(y)+1))

plt.plot(x, y)
# 保存图片
plt.savefig("./t1.png")
# 展示图片
plt.show()
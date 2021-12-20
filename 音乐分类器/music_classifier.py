from pprint import pprint
import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.io import wavfile
# from scipy import fft

# 用于保存模块
import pickle

'''
# 读取傅里叶变换之后的数据集，将其做成机器学习所需要的X和y
genre_list = ['classical', 'metal', 'country', 'pop', 'rock', 'jazz']
X = []
y = []
for g in genre_list:
    for n in range(100):
        rad = "C:/Users/10509/Desktop/AIStudy/音乐分类器/数据/"+g+"."+str(n).zfill(5)+".fft.npy"
        fft_features = np.load(rad)
        X.append(fft_features)
        y.append(genre_list.index(g))

X = np.array(X)
y = np.array(y)

#model = LogisticRegression(multi_class="multinomial", solver='sag', max_iter=10000)
model = LogisticRegression()
model.fit(X, y)

output = open('model.pkl', 'wb')
pickle.dump(model, output)
output.close()
'''


pkl_file = open('model.pkl', 'rb')
model_loaded = pickle.load(pkl_file)
pprint(model_loaded)
#
pkl_file.close()

print('Starting read wavfile...')
music_name = "邓紫棋 - 喜欢你.wav"
sample_rate, X = wavfile.read("C:/Users/10509/Desktop/AIStudy/音乐分类器/数据/genres/邓紫棋 - 喜欢你.wav")

# 将双通道歌曲reshape单通道
X = np.reshape(X, (1, -1))[0]
print(X.shape)
test_fft_features = abs(np.fft.fft(X)[:1000])
print(sample_rate, test_fft_features, len(test_fft_features))

temp = model_loaded.predict([test_fft_features])
print(temp)
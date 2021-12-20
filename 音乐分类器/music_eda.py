from scipy import fft
from scipy.io import wavfile
from matplotlib.pyplot import specgram
import matplotlib.pyplot as plt
import numpy as np


# 准备音乐数据，把音乐文件一个一个的去使用傅里叶变换，并且把傅里叶变换之后的结果落地保存。
# 提取特征

def create_fft(g, n):
    rad = "C:/Users/10509/Desktop/AIStudy/音乐分类器/数据/genres/"+g+"/converted/"+g+"."+str(n).zfill(5)+".au.wav"
    sample_rate, X = wavfile.read(rad)
    # 保证每首歌X都是1:1000
    fft_features = abs(np.fft.fft(X)[:1000])
    sad = "C:/Users/10509/Desktop/AIStudy/音乐分类器/数据/"+g+"."+str(n).zfill(5)+".fft"
    np.save(sad, fft_features)

genre_list = ['classical', 'jazz', 'country', 'pop', 'rock', 'metal']
for g in genre_list:
    for n in range(100):
        create_fft(g, n)
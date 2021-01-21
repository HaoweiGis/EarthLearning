'''
Author: WYN
Date: 2020-11-12 20:47:13
LastEditors: WYN
LastEditTime: 2020-11-28 16:35:24
'''
#导入模块
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
#导入scipy模块
from scipy import stats
from scipy.stats import shapiro
import scipy
from scipy import stats


#构造一组随机数据
# df=pd.read_csv(r'C:\Users\Wen\Desktop\hlj\ratio10.csv')
# df=pd.read_csv(r'C:\Users\Wen\Desktop\hlj\vh100.csv')
# df=pd.read_csv(r'C:\Users\Wen\Desktop\hlj\vv1000.csv')
# df=pd.read_csv(r'C:\Users\Wen\Desktop\hlj\subtract100.csv')
# df=pd.read_csv(r'C:\Users\Wen\Desktop\hlj\b2341000.csv')
# df=pd.read_csv(r'C:\Users\Wen\Desktop\hlj\ndre10.csv')
# df=pd.read_csv(r'C:\Users\Wen\Desktop\hlj\ndre-1000.csv')
# df=pd.read_csv(r'C:\Users\Wen\Desktop\hlj\ndre1100.csv')
# df=pd.read_csv(r'C:\Users\Wen\Desktop\hlj\ndre21000.csv')
# df=pd.read_csv(r'C:\Users\Wen\Desktop\hlj\ndre31000.csv')
# df=pd.read_csv(r'C:\Users\Wen\Desktop\hlj\b8a1000.csv')
# df=pd.read_csv(r'C:\Users\Wen\Desktop\hlj\b111000.csv')
df=pd.read_csv(r'C:\Users\Wen\Desktop\11.csv')

s= np.array(df.iloc[:,[0]]).ravel()
# s= np.array(df.iloc[:,[1]]).ravel()
# s= np.array(df.iloc[:,[2]]).ravel()
# s= np.array(df.iloc[:,[3]]).ravel()
# s= np.array(df.iloc[:,[4]]).ravel()
# s= np.array(df.iloc[:,[5]]).ravel()
# s= np.array(df.iloc[:,[6]]).ravel()
# print(s)
# print(s2)
# print(s3)
# print(s)

"""
kstest方法：KS检验，参数分别是：待检验的数据，检验方法（这里设置成norm正态分布），均值与标准差
结果返回两个值：statistic → D值，pvalue → P值
p值大于0.05，为正态分布
H0:样本符合  
H1:样本不符合 
如何p>0.05接受H0 ,反之 
"""

u = s.mean()  # 计算均值
std = s.std()  # 计算标准差
ks = stats.kstest(s, 'norm', (u, std))
skew = stats.skew(s)
kurtosis = stats.kurtosis(s)

print('均值：',u)
print('标准差：',std)
print('峰度：',kurtosis)
print('偏度：',skew)
print('ks检验：',ks)

# kse = scipy.stats.anderson(s,dist='norm')
# print(kse)

sh = shapiro(s)
print('sw检验：',sh)
# result1
from os import close
from numpy.lib.npyio import NpzFile
import pandas as pd
import os
import tqdm
import re
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy import stats
from sklearn.linear_model import TheilSenRegressor

SaveFile_Path =  r'C:\Users\hp\Desktop\wdpashp\Summary\result1'       #拼接后要保存的文件路径     
Base_path = r'‪C:\Users\hp\Desktop\wdpashp\Summary\result1\case1sen.csv'.replace('\u202a','')

# 从空间抽取数据
def SpaticalSelect(filename):
    basedf = pd.read_csv(Base_path)
    # viirs2018
    # dmsp2010
    df = pd.read_csv(os.path.join(SaveFile_Path,'buffer0.csv'))
    basedf = basedf.merge(df, on='objectid',how='left')
    basedf.to_csv(os.path.join(SaveFile_Path,filename), index=False)

def LinearModel(x,y):
    model = LinearRegression()
    model.fit(x, y)
    y_pred = model.predict(x)
    a = model.coef_
    b = model.intercept_
    r2 = r2_score(y,y_pred)
    return y_pred,a,b,r2

def writefile(filename,lines):
    f = open(filename,'a')
    for line in lines:
        f.writelines(line + '\n')

if __name__ == "__main__":

    # SpaticalSelect('case1_buffer0.csv')


    # part IV 计算最大buffer的平均值的斜率
    df= pd.read_csv(os.path.join(SaveFile_Path,'case1_buffer0.csv'))
    ntlValue = np.array(df.iloc[:,1:])
    ids = np.array(df.iloc[:,[0]]).ravel()
    firstArr = None
    lines = []
    for i in tqdm.tqdm(range(len(ids))):
        x = np.arange(1, 28, 1)
        x_lable = [str(i) for i in np.arange(1992, 2019, 1)]
        y = ntlValue[i,:]
        x_trans = x.reshape(-1, 1)
        y_trans = y.reshape(-1, 1)
        y1,a1,b1,r2_1 = LinearModel(x_trans,y_trans)
        tau, p_value = stats.kendalltau(x_trans, y_trans)
        res = stats.theilslopes(y_trans,x_trans , 0.90)
        r2_res = r2_score(y_trans,res[1] + res[0] * x_trans)

        if int(ids[i]) == 44970 or int(ids[i]) == 41540 or int(ids[i]) == 30906 or int(ids[i]) == 34297:
            plt.rcParams['axes.facecolor'] = '#424242'
            plt.rc('font',family='Times New Roman')
            plt.figure(figsize=(13,10))
            # plt.plot(x.reshape(-1, 1), y1, 'g-')

            plt.xlabel('Year',size = 40,weight="bold")
            plt.ylabel('NTL DN Value', size = 40,weight="bold")
            plt.xticks(x, x_lable, fontproperties = 'Times New Roman', size = 35,weight="bold") 
            plt.yticks(fontproperties = 'Times New Roman', size = 35,weight="bold")
            # plt.ylim(10, 30)

            ax= plt.gca()
            for tick in ax.get_xticklabels():
                tick.set_rotation(45)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            import matplotlib.ticker as ticker
            ax.xaxis.set_major_locator(ticker.MultipleLocator(3))
            # ax.yaxis.set_major_locator(ticker.MultipleLocator(5))

            plt.scatter(x, y, s=200 ,c='#B36E5F')
            plt.plot(x, res[1] + res[0] * x, '-',color='#E3B96D',linewidth=3)
            # plt.plot(x, res[1] + res[2] * x, 'r--')
            # plt.plot(x, res[1] + res[3] * x, 'r--')

            plt.title("max_index:"+str(ids[i]),size = 15)
            line_plot = str(ids[i]) + ',' + str(a1) + ',' + str(b1) + ',' + str(r2_1)  + ',' +str(tau) + ',' + str(p_value) + ',' + str(res[0]) + ',' + str(res[1])  + ',' + str(res[2]) + ',' + str(res[3]) + ',' + str(r2_res)
            print(line_plot)
            plt.tight_layout()
            plt.show()
            # plt.savefig(os.path.join(SaveFile_Path_sum_base ,'Picture1',str(int(ids[i])) + '.png'))
            plt.close()

        line = str(ids[i]) + ',' + str(a1) + ',' + str(b1) + ',' + str(r2_1)  + ',' +str(tau) + ',' + str(p_value) + ',' + str(res[0]) + ',' + str(res[1])  + ',' + str(res[2]) + ',' + str(res[3]) + ',' + str(r2_res)

    #     lines.append(line.replace('[','').replace(']',''))
    # writefile(os.path.join(SaveFile_Path, 'buffer0Slopsen_new.csv'),lines)
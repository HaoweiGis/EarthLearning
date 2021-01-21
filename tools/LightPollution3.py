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

Folder_Path = r'C:\Users\hp\Desktop\wdpashp\wdpadata1\wdpa_ntl7_count'          #要拼接的文件夹及其完整路径，注意不要包含中文

SaveFile_Path_coresum =  r'C:\Users\hp\Desktop\wdpashp\wdpadata3\Picuture\case\BufferSum'
SaveFile_Path_sum =  r'C:\Users\hp\Desktop\wdpashp\wdpadata3\Picuture\case\RingBufferSum'
SaveFile_Path_sum_base =  r'C:\Users\hp\Desktop\wdpashp\wdpadata3\Picuture\case\intensityrange\RingBufferSum_base'
# Base_path = r'C:\Users\hp\Desktop\wdpashp\wdpadata3\Picuture\case\case.csv'
Base_path = r'C:\Users\hp\Desktop\wdpashp\wdpadata3\Picuture\case\case2.csv'
# Base_path = r'C:\Users\hp\Desktop\wdpashp\wdpadata3\Picuture\case\RingBufferSum\bufferIntensityMode.csv'
Base_path_First = r'C:\Users\hp\Desktop\wdpashp\wdpadata3\Picuture\case\intensityrange\RingIntensityRange.csv'

Base_path_Core = r'C:\Users\hp\Desktop\wdpashp\wdpadata3\Picuture\case\intensityrange\RingBufferSum_base\CorePollution_base.csv'
SaveFile_Path_sum_base_core =  r'C:\Users\hp\Desktop\wdpashp\wdpadata3\Picuture\case\intensityrange\corePollution'

def getRingSumValue_core(filename,year):
    df= pd.read_csv(os.path.join(SaveFile_Path_coresum,filename))
    ntlValue = np.array(df.iloc[:,1:])
    firstArr = None
    ids = np.array(df.iloc[:,[0]]).ravel()
    for i in range(len(ids)):
        value = ntlValue[i,0]
        id = ids[i]
        firstLine = np.array([id,value])
        if firstArr is None:
            firstArr = firstLine
        else:
            firstArr = np.vstack((firstArr,firstLine)) 
    pdValue = pd.DataFrame(firstArr,columns=['objectid',year])
    return pdValue

def getRingSumValue(filename,year):
    df= pd.read_csv(os.path.join(SaveFile_Path_sum_base,filename))
    ntlValue = np.array(df.iloc[:,1:])
    firstArr = None
    ids = np.array(df.iloc[:,[0]]).ravel()
    for i in range(len(ids)):
        value = ntlValue[i,2:]
        id = ids[i]
        indexmin = int(ntlValue[i,0])-1
        indexmax = int(ntlValue[i,1])
        value[np.where(np.isinf(value))] = np.nan
        if indexmax == indexmin:
            meanV = value[indexmin-1]
        else:
            meanV = np.nansum(value[indexmin:indexmax])/(indexmax-indexmin)
        # if id == 16515:
        #     gouzi
        firstLine = np.array([id,meanV])
        if firstArr is None:
            firstArr = firstLine
        else:
            firstArr = np.vstack((firstArr,firstLine)) 
    pdValue = pd.DataFrame(firstArr,columns=['objectid',year])
    return pdValue

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

    # # part 合并BaseDF
    # for year in tqdm.tqdm(range(1992,2019,1)):
    #     filename = 'case' + str(year) + '.csv'
    #     basedf = pd.read_csv(Base_path_First)
    #     df= pd.read_csv(os.path.join(SaveFile_Path_sum,filename))
    #     basedf = basedf.merge(df, on='objectid',how='left')
    #     basedf.to_csv(os.path.join(SaveFile_Path_sum_base,filename), index=False)

    # # # part II 获取1992-2018年灯光最强的index
    # ValueList = []
    # for year in tqdm.tqdm(range(1992,2019,1)):
    #     filename = 'case' + str(year) + '.csv'
    #     pdValue = getRingSumValue(filename,str(year))
    #     ValueList.append(pdValue)
    # basedf = None
    # for value in ValueList:
    #     if basedf is None:
    #         basedf = value
    #     else:
    #         basedf = basedf.merge(value, on='objectid',how='left')
    # basedf.to_csv(os.path.join(SaveFile_Path_sum_base,'IntensitybufferSum1.csv'), index=False)

    # # part IV 计算最大buffer的众数的斜率
    # df= pd.read_csv(os.path.join(SaveFile_Path_sum_base,'IntensitybufferSum1.csv'))
    # ntlValue = np.array(df.iloc[:,1:])
    # ids = np.array(df.iloc[:,[0]]).ravel()
    # firstArr = None
    # lines = []
    # for i in tqdm.tqdm(range(len(ids))):
    # #     x = np.arange(1, 28, 1)
    # #     x_lable = [str(i) for i in np.arange(1992, 2019, 1)]
    # #     y = ntlValue[i,:]
    # #     y1,a1,b1,r2_1 = LinearModel(x.reshape(-1, 1),y.reshape(-1, 1))
    # #     tau, p_value = stats.kendalltau(x, y)
    # #     res = stats.theilslopes(y, x, 0.90)
    # #     # print(res[0],res[1],res[2],res[3])
    # #     r2_res = r2_score(y,res[1] + res[0] * x)
    # #     line = str(ids[i]) + ',' + str(a1) + ',' + str(b1) + ',' + str(r2_1) + ',' +str(tau) + ',' + str(p_value) + ',' + str(res[0]) + ',' + str(res[1])  + ',' + str(res[2]) + ',' + str(res[3]) + ',' + str(r2_res)
    # #     lines.append(line.replace('[','').replace(']',''))
    # # writefile(os.path.join(SaveFile_Path_sum_base, 'IntensitybufferSumslopsen1.csv'),lines)

        
    #     if int(ids[i]) == 13319:
    #         x = np.arange(1, 28, 1)
    #         x_lable = [str(i) for i in np.arange(1992, 2019, 1)]
    #         y = ntlValue[i,:]
    #         y1,a1,b1,r2_1 = LinearModel(x.reshape(-1, 1),y.reshape(-1, 1))
    #         tau, p_value = stats.kendalltau(x, y)
    #         res = stats.theilslopes(y, x, 0.90)
    #         # print(res[0],res[1],res[2],res[3])
    #         r2_res = r2_score(y,res[1] + res[0] * x)

    #         plt.rcParams['axes.facecolor'] = '#424242'
    #         # plt.patch.set_alpha(0.)
    #         plt.rc('font',family='Times New Roman')
    #         plt.figure(figsize=(18,4))
    #         # plt.plot(x.reshape(-1, 1), y1, 'g-')

    #         plt.xlabel('Year',size = 35,color= '#FFFFFF')
    #         plt.ylabel('NTL DN Value', size = 35,color= '#FFFFFF')
    #         plt.xticks(x, x_lable, fontproperties = 'Times New Roman', size = 30,color= '#FFFFFF') 
    #         plt.yticks(fontproperties = 'Times New Roman', size = 30,color= '#FFFFFF')

    #         ax= plt.gca()
    #         for tick in ax.get_xticklabels():
    #             tick.set_rotation(45)
    #         for tick in ax.get_yticklabels():
    #             tick.set_rotation(90)

    #         ax.spines['bottom'].set_linewidth(3);###设置底部坐标轴的粗细
    #         ax.spines['left'].set_linewidth(3);####设置左边坐标轴的粗细
    #         ax.spines['top'].set_visible(False)
    #         ax.spines['right'].set_visible(False)
    #         import matplotlib.ticker as ticker
    #         ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    #         ax.yaxis.set_major_locator(ticker.MultipleLocator(50))

    #         plt.scatter(x, y, s=300 ,c='#6930C3',edgecolors='w',linewidths=2)
    #         plt.plot(x, res[1] + res[0] * x, '-',color='#80FFDB',linewidth=5)
    #         # plt.plot(x, res[1] + res[2] * x, 'r--')
    #         # plt.plot(x, res[1] + res[3] * x, 'r--')

    #         # plt.title("max_index:"+str(ids[i]),size = 15)
    #         line = str(ids[i]) + ',' + str(a1) + ',' + str(b1) + ',' + str(r2_1) + ',' +str(tau) + ',' + str(p_value) + ',' + str(res[0]) + ',' + str(res[1])  + ',' + str(res[2]) + ',' + str(res[3]) + ',' + str(r2_res)
    #         print(line)
    #         plt.tight_layout()
    #         plt.savefig(os.path.join(SaveFile_Path_sum_base ,'Picture',str(int(ids[i])) + '1.png'),transparent=True)
    #         plt.show()
    #         plt.close()


    # # # part II 获取1992-2018年灯光最强的index Core Pollution
    # ValueList = []
    # for year in tqdm.tqdm(range(1992,2019,1)):
    #     filename = 'case_' + str(year) + '.csv'
    #     pdValue = getRingSumValue_core(filename,str(year))
    #     ValueList.append(pdValue)

    # basedf = pd.read_csv(Base_path_Core)
    # for value in ValueList:
    #     if basedf is None:
    #         basedf = value
    #     else:
    #         basedf = basedf.merge(value, on='objectid',how='left')
    # basedf.to_csv(os.path.join(SaveFile_Path_sum_base_core,'IntensitybufferSum_core.csv'), index=False)

    # part IV 计算最大buffer的众数的斜率 Core Pollution
    df= pd.read_csv(os.path.join(SaveFile_Path_sum_base_core,'IntensitybufferSum_core.csv'))
    ntlValue = np.array(df.iloc[:,1:])
    ids = np.array(df.iloc[:,[0]]).ravel()
    firstArr = None
    lines = []
    for i in tqdm.tqdm(range(len(ids))):
        x = np.arange(1, 28, 1)
        x_lable = [str(i) for i in np.arange(1992, 2019, 1)]
        y = ntlValue[i,:]
        y1,a1,b1,r2_1 = LinearModel(x.reshape(-1, 1),y.reshape(-1, 1))
        tau, p_value = stats.kendalltau(x, y)
        res = stats.theilslopes(y, x, 0.90)
        # print(res[0],res[1],res[2],res[3])
        r2_res = r2_score(y,res[1] + res[0] * x)

        line = str(ids[i]) + ',' + str(a1) + ',' + str(b1) + ',' + str(r2_1) + ',' +str(tau) + ',' + str(p_value) + ',' + str(res[0]) + ',' + str(res[1])  + ',' + str(res[2]) + ',' + str(res[3]) + ',' + str(r2_res)
        lines.append(line.replace('[','').replace(']',''))
    writefile(os.path.join(SaveFile_Path_sum_base, 'IntensitybufferSumslopsen1.csv'),lines)

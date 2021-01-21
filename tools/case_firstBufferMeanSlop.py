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

Folder_Path = r'C:\Users\hp\Desktop\wdpashp\wdpadata1\wdpa_ntl7'          #要拼接的文件夹及其完整路径，注意不要包含中文
SaveFile_Path =  r'C:\Users\hp\Desktop\wdpashp\wdpadata3\Picuture\case\AllBuffer_case2_mean'       #拼接后要保存的文件路径     
# Base_path = r'C:\Users\hp\Desktop\wdpashp\wdpadata3\Picuture\case\case2.csv'
Base_path = r'C:\Users\hp\Desktop\wdpashp\wdpadata3\Picuture\case\AllBuffer_case2_mean\FirstMaxValue_base.csv'

# 从空间抽取数据
def SpaticalSelect(pattern,filename):
    basedf = pd.read_csv(Base_path)
    # viirs2018
    # dmsp2010
    for i in range(0,51,1):
        df = pd.read_csv(os.path.join(Folder_Path,'wdpa_ntl7_year_buffer'+str(i)+'.csv'))
        df = df[['objectid',pattern]]
        basedf = basedf.merge(df, on='objectid',how='left',suffixes = ('','_b'+str(i)))
        # .fillna("")
    basedf.to_csv(os.path.join(SaveFile_Path,filename), index=False)

# # 从时间上抽取数据
# df= pd.read_csv(os.path.join(SaveFile_Path,"wdpa_ntl7_year_buffer50.csv"))
# patterndmsp = re.compile('^dmsp(\d)+$')
# patternviirs = re.compile('^viirs(\d)+')
# col1 = [i for i in df.columns if patterndmsp.match(i)]
# col2 = [i for i in df.columns if patternviirs.match(i)]
# col = col1+ col2 + ['objectid']
# df = df[col]
# df.to_csv(os.path.join(SaveFile_Path,"buffer50.csv"), index=False)

def getFirstValue(filename,year):
    df= pd.read_csv(os.path.join(SaveFile_Path,filename))
    ntlValue = np.array(df.iloc[:,1:])
    firstArr = None
    ids = np.array(df.iloc[:,[0]]).ravel()
    for i in range(len(ids)):
        value = ntlValue[i,:]
        id = ids[i]
        first = value[np.isnan(value)].shape[0]
        # if id == 27830:
        #     print(fdaf)
        firstLine = np.array([id,first])
        if firstArr is None:
            firstArr = firstLine
        else:
            firstArr = np.vstack((firstArr,firstLine)) 
    pdValue = pd.DataFrame(firstArr,columns=['objectid',year])
    return pdValue

def getFirstMaxValueMean(filename,year):
    df= pd.read_csv(os.path.join(SaveFile_Path,filename))
    ntlValue = np.array(df.iloc[:,1:])
    firstArr = None
    ids = np.array(df.iloc[:,[0]]).ravel()
    for i in range(len(ids)):
        value = ntlValue[i,1:]
        max = int(ntlValue[i,0])
        id = ids[i]
        # if id == 27830:
        #     print(fdaf)
        gouzi = value[:max+1]
        ntlmean = np.nanmean(value[:max+1])
        firstLine = np.array([id,ntlmean])
        if firstArr is None:
            firstArr = firstLine
        else:
            firstArr = np.vstack((firstArr,firstLine)) 

    pdValue = pd.DataFrame(firstArr,columns=['objectid',year])
    return pdValue


def getFirstMaxValue(filename,year):
    df= pd.read_csv(os.path.join(SaveFile_Path,filename))
    ntlValue = np.array(df.iloc[:,1:])
    firstArr = None
    ids = np.array(df.iloc[:,[0]]).ravel()
    for i in range(len(ids)):
        value = ntlValue[i,1:]
        max = int(ntlValue[i,0])
        id = ids[i]
        # if id == 27830:
        #     print(fdaf)
        meanV = value[max]
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
    # # part I and partII
    # for year in tqdm.tqdm(range(1992,2014,1)):
    #     pattern = 'dmsp' + str(year)
    #     filename = 'case_' + str(year) + '.csv'
    #     SpaticalSelect(pattern,filename)
    # for year in tqdm.tqdm(range(2014,2019,1)):
    #     pattern = 'viirs' + str(year)
    #     filename = 'case_' + str(year) + '.csv'
    #     SpaticalSelect(pattern,filename)

    # # part II 获取1992-2018年首先出现灯光的缓冲区
    # ValueList = []
    # for year in tqdm.tqdm(range(1992,2019,1)):
    #     filename = 'case_' + str(year) + '.csv'
    #     pdValue = getFirstValue(filename,str(year))
    #     ValueList.append(pdValue)
    # basedf = None
    # for value in ValueList:
    #     if basedf is None:
    #         basedf = value
    #     else:
    #         basedf = basedf.merge(value, on='objectid',how='left')
    # basedf.to_csv(os.path.join(SaveFile_Path,'FirstYear.csv'), index=False)

    # # part III 获取1992-2018年首先出现灯光的缓冲区最大buffer的平均值的时间序列变化
    # ValueList = []
    # for year in tqdm.tqdm(range(1992,2019,1)):
    #     filename = 'case_' + str(year) + '.csv'
    #     pdValue = getFirstMaxValueMean(filename,str(year))
    #     ValueList.append(pdValue)
    # basedf = None
    # for value in ValueList:
    #     if basedf is None:
    #         basedf = value
    #     else:
    #         basedf = basedf.merge(value, on='objectid',how='left')
    # basedf.to_csv(os.path.join(SaveFile_Path,'bufferMean.csv'), index=False)

    # # part III 获取1992-2018年首先出现灯光的缓冲区最大buffer的值
    # ValueList = []
    # for year in tqdm.tqdm(range(1992,2019,1)):
    #     filename = 'case_' + str(year) + '.csv'
    #     pdValue = getFirstMaxValue(filename,str(year))
    #     ValueList.append(pdValue)
    # basedf = None
    # for value in ValueList:
    #     if basedf is None:
    #         basedf = value
    #     else:
    #         basedf = basedf.merge(value, on='objectid',how='left')
    # basedf.to_csv(os.path.join(SaveFile_Path,'bufferMeanValue.csv'), index=False)

    # part IV 计算最大buffer的平均值的斜率
    df= pd.read_csv(os.path.join(SaveFile_Path,'bufferMeanValue.csv'))
    ntlValue = np.array(df.iloc[:,1:])
    ids = np.array(df.iloc[:,[0]]).ravel()
    firstArr = None
    lines = []
    for i in tqdm.tqdm(range(len(ids))):
        x = np.arange(1992, 2019, 1)
        y = ntlValue[i,:]
        y1,a1,b1,r2_1 = LinearModel(x.reshape(-1, 1),y.reshape(-1, 1))
        tau, p_value = stats.kendalltau(x, y)
        res = stats.theilslopes(y, x, 0.90)
        if int(ids[i]) == 13319:
            plt.plot(x, y, 'k.')
            plt.plot(x.reshape(-1, 1), y1, 'g-')
            plt.title("max_index:"+str(ids[i]),size = 15)
            # plt.show()
            plt.savefig(os.path.join(SaveFile_Path ,'PictureValue',str(int(ids[i])) + '.png'))
            plt.close()
            print('13319 is OK')
        line = str(ids[i]) + ',' + str(a1) + ',' + str(b1) + ',' + str(r2_1) + ',' +str(tau) + ',' + str(p_value) + ',' + str(res[0]) + ',' + str(res[1])  + ',' + str(res[2]) + ',' + str(res[3])
        lines.append(line.replace('[','').replace(']',''))
    writefile(os.path.join(SaveFile_Path, 'bufferMeanSlopValue2018sen.csv'),lines)

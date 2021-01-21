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

Folder_Path = r'C:\Users\hp\Desktop\wdpashp\wdpadata1\wdpa_ntl7_ring_new'          #要拼接的文件夹及其完整路径，注意不要包含中文
SaveFile_Path =  r'C:\Users\hp\Desktop\wdpashp\wdpadata3\Picuture\case\RingBufferMode'       #拼接后要保存的文件路径     
# Base_path = r'C:\Users\hp\Desktop\wdpashp\wdpadata3\Picuture\case\case.csv'
Base_path = r'C:\Users\hp\Desktop\wdpashp\wdpadata3\Picuture\case\RingBuffer\intensityMode.csv'

# 从空间抽取数据
def SpaticalSelect(pattern,filename):
    basedf = pd.read_csv(Base_path)
    # viirs2018
    # dmsp2010
    for i in range(1,51,1):
        df = pd.read_csv(os.path.join(Folder_Path,'wdpa_ntl7_ring'+str(i)+'.csv'))
        if pattern in df.columns:
            df = df[['objectid',pattern]]
        else:
            df[pattern]=np.nan
            df = df[['objectid',pattern]]
        basedf = basedf.merge(df, on='objectid',how='left',suffixes = ('','_b'+str(i)))
        # .fillna("")
    basedf.to_csv(os.path.join(SaveFile_Path,filename), index=False)

def getFirstValue(filename,year):
    df= pd.read_csv(os.path.join(SaveFile_Path,filename))
    ntlValue = np.array(df.iloc[:,1:])
    firstArr = None
    ids = np.array(df.iloc[:,[0]]).ravel()
    for i in range(len(ids)):
        value = ntlValue[i,:]
        id = ids[i]
        indexArr = np.arange(1,51,1)
        intensity = np.divide(value,indexArr)
        intensitymax = np.where(intensity==np.nanmax(intensity))[0][0]
        firstLine = np.array([id,intensitymax])
        if firstArr is None:
            firstArr = firstLine
        else:
            firstArr = np.vstack((firstArr,firstLine)) 

    pdValue = pd.DataFrame(firstArr,columns=['objectid',year])
    return pdValue

def getIntensityValue(filename,year):
    df= pd.read_csv(os.path.join(SaveFile_Path,filename))
    ntlValue = np.array(df.iloc[:,1:])
    firstArr = None
    ids = np.array(df.iloc[:,[0]]).ravel()
    for i in range(len(ids)):
        value = ntlValue[i,1:]
        mode = int(ntlValue[i,0])
        id = ids[i]
        # if id == 27830:
        #     print(fdaf)
        modeV = value[mode]
        firstLine = np.array([id,modeV])
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
    # # part I and partII(替换BaseDF)
    # for year in tqdm.tqdm(range(1992,2014,1)):
    #     pattern = 'dmsp' + str(year)
    #     filename = 'case_' + str(year) + '.csv'
    #     SpaticalSelect(pattern,filename)
    # for year in tqdm.tqdm(range(2014,2019,1)):
    #     pattern = 'viirs' + str(year)
    #     filename = 'case_' + str(year) + '.csv'
    #     SpaticalSelect(pattern,filename)

    # # part II 获取1992-2018年灯光最强的index
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
    # basedf.to_csv(os.path.join(SaveFile_Path,'bufferIntensity.csv'), index=False)

    # # part II 获取1992-2018年灯光最强的index 众数的值
    ValueList = []
    for year in tqdm.tqdm(range(1992,2019,1)):
        filename = 'case_' + str(year) + '.csv'
        pdValue = getIntensityValue(filename,str(year))
        ValueList.append(pdValue)
    basedf = None
    for value in ValueList:
        if basedf is None:
            basedf = value
        else:
            basedf = basedf.merge(value, on='objectid',how='left')
    basedf.to_csv(os.path.join(SaveFile_Path,'bufferIntensityMode.csv'), index=False)

    # # part IV 计算最大buffer的众数的斜率
    # df= pd.read_csv(os.path.join(SaveFile_Path,'bufferMean.csv'))
    # ntlValue = np.array(df.iloc[:,1:])
    # ids = np.array(df.iloc[:,[0]]).ravel()
    # firstArr = None
    # lines = []
    # for i in tqdm.tqdm(range(len(ids))):
    #     x = np.arange(1992, 2014, 1)
    #     y = ntlValue[i,:-5]
    #     y1,a1,b1,r2_1 = LinearModel(x.reshape(-1, 1),y.reshape(-1, 1))
    #     if int(ids[i]) == 13319:
    #         plt.plot(x, y, 'k.')
    #         plt.plot(x.reshape(-1, 1), y1, 'g-')
    #         plt.title("max_index:"+str(ids[i]),size = 15)
    #         # plt.show()
    #         plt.savefig(os.path.join(SaveFile_Path ,'Picture1',str(int(ids[i])) + '.png'))
    #         plt.close()
    #         print('13319 is OK')
    #     line = str(ids[i]) + ',' + str(a1) + ',' + str(b1) + ',' + str(r2_1)
    #     lines.append(line.replace('[','').replace(']',''))
    # writefile(os.path.join(SaveFile_Path, 'bufferMeanSlop2013.csv'),lines) 

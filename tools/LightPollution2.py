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
SaveFile_Path_base =  r'C:\Users\hp\Desktop\wdpashp\wdpadata3\Picuture\case\BufferCount'    
SaveFile_Path =  r'C:\Users\hp\Desktop\wdpashp\wdpadata3\Picuture\case\RingBufferCount'       #拼接后要保存的文件路径     
SaveFile_Path_sum =  r'C:\Users\hp\Desktop\wdpashp\wdpadata3\Picuture\case\RingBufferSum'
SaveFile_Path_mean =  r'C:\Users\hp\Desktop\wdpashp\wdpadata3\Picuture\case\RingBufferMean'
SaveFile_Path_mean_base =  r'C:\Users\hp\Desktop\wdpashp\wdpadata3\Picuture\case\RingBufferMean_base'
SaveFile_Path_sum_base =  r'C:\Users\hp\Desktop\wdpashp\wdpadata3\Picuture\case\RingBufferSum_base'
# Base_path = r'C:\Users\hp\Desktop\wdpashp\wdpadata3\Picuture\case\case.csv'
Base_path = r'C:\Users\hp\Desktop\wdpashp\wdpadata3\Picuture\case\case2.csv'
# Base_path = r'C:\Users\hp\Desktop\wdpashp\wdpadata3\Picuture\case\RingBufferSum\bufferIntensityMode.csv'
Base_path_First = r'C:\Users\hp\Desktop\wdpashp\wdpadata3\Picuture\case\RingBufferMean\FirstRingBase.csv'

# 从空间抽取数据
def SpaticalSelect(pattern,filename):
    basedf = pd.read_csv(Base_path)
    # viirs2018
    # dmsp2010
    for i in range(0,51,1):
        df = pd.read_csv(os.path.join(Folder_Path,'wdpa_ntl7_count'+str(i)+'.csv'))
        if pattern in df.columns:
            df = df[['objectid',pattern]]
        else:
            df[pattern]=np.nan
            df = df[['objectid',pattern]]
        basedf = basedf.merge(df, on='objectid',how='left',suffixes = ('','_b'+str(i)))
        # .fillna("")
    basedf.to_csv(os.path.join(SaveFile_Path,filename), index=False)


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
            goubi = value[indexmin:indexmax]
            meanV = np.nansum(value[indexmin:indexmax])
        # if id == 16515:
        #     gouzi
        firstLine = np.array([id,meanV])
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
        value = ntlValue[i,2:]
        id = ids[i]
        mode = int(ntlValue[i,0])
        value_before = ntlValue[i,1:-1]
        ringV = value - value_before
        indexArr = np.arange(1,51,1)
        intensity = np.divide(ringV,indexArr)
        modeV = intensity[mode]
        firstLine = np.array([id,modeV])
        if firstArr is None:
            firstArr = firstLine
        else:
            firstArr = np.vstack((firstArr,firstLine)) 
    pdValue = pd.DataFrame(firstArr,columns=['objectid',year])
    return pdValue

def getRingSumValue2csv(filename,year):
    df= pd.read_csv(os.path.join(SaveFile_Path_base,filename))
    columnsArr = np.delete(df.columns,1)
    ntlValue = np.array(df.iloc[:,1:])
    firstArr = None
    ids = np.array(df.iloc[:,[0]]).ravel()
    for i in range(len(ids)):
        value = ntlValue[i,1:]
        id = ids[i]
        value_before = ntlValue[i,:-1]
        ringV = value - value_before
        firstLine = np.hstack((id,ringV))
        if firstArr is None:
            firstArr = firstLine
        else:
            firstArr = np.vstack((firstArr,firstLine)) 
    pdValue = pd.DataFrame(firstArr,columns=columnsArr)
    return pdValue


def getRingMeanValue2csv(filename,year):
    df= pd.read_csv(os.path.join(SaveFile_Path,filename))
    columnsArr = df.columns
    ntlValue = np.array(df.iloc[:,1:])

    df_sum= pd.read_csv(os.path.join(SaveFile_Path_sum,filename))

    firstArr = None
    ids = np.array(df.iloc[:,[0]]).ravel()
    for i in range(len(ids)):
        value = ntlValue[i,:]
        id = ids[i]
        value_sum = np.array(df_sum.loc[df_sum['objectid'] == id]).ravel()[1:]
        meanV = value_sum/value
        firstLine = np.hstack((id,meanV))
        if firstArr is None:
            firstArr = firstLine
        else:
            firstArr = np.vstack((firstArr,firstLine)) 
    pdValue = pd.DataFrame(firstArr,columns=columnsArr)
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

    # # part II 获取ring buffer
    # for year in tqdm.tqdm(range(1992,2019,1)):
    #     filename = 'case_' + str(year) + '.csv'
    #     pdValue = getRingSumValue2csv(filename,str(year))
    #     pdValue.to_csv(os.path.join(SaveFile_Path,'case' + str(year) + '.csv'), index=False)

    # # part II 获取ring buffer Mean
    # for year in tqdm.tqdm(range(1992,2019,1)):
    #     filename = 'case' + str(year) + '.csv'
    #     pdValue = getRingMeanValue2csv(filename,str(year))
    #     pdValue.to_csv(os.path.join(SaveFile_Path_mean,'case' + str(year) + '.csv'), index=False)


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
    # basedf.to_csv(os.path.join(SaveFile_Path_sum_base,'bufferSum.csv'), index=False)

    # part IV 计算最大buffer的众数的斜率
    df= pd.read_csv(os.path.join(SaveFile_Path_sum_base,'bufferSum.csv'))
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
        
        if int(ids[i]) == 40780:
            plt.rcParams['axes.facecolor'] = '#424242'
            plt.rc('font',family='Times New Roman')
            plt.figure(figsize=(15,8))
            # plt.plot(x.reshape(-1, 1), y1, 'g-')

            plt.xlabel('Year',size = 35)
            plt.ylabel('NTL DN Value', size = 35)
            plt.xticks(x, x_lable, fontproperties = 'Times New Roman', size = 30) 
            plt.yticks(fontproperties = 'Times New Roman', size = 30)

            ax= plt.gca()
            for tick in ax.get_xticklabels():
                tick.set_rotation(45)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            import matplotlib.ticker as ticker
            ax.xaxis.set_major_locator(ticker.MultipleLocator(2))

            plt.scatter(x, y, s=200 ,c='#B36E5F')
            plt.plot(x, res[1] + res[0] * x, '-',color='#E3B96D',linewidth=3)
            # plt.plot(x, res[1] + res[2] * x, 'r--')
            # plt.plot(x, res[1] + res[3] * x, 'r--')

            plt.title("max_index:"+str(ids[i]),size = 15)
            line = str(ids[i]) + ',' + str(a1) + ',' + str(b1) + ',' + str(r2_1) + ',' +str(tau) + ',' + str(p_value) + ',' + str(res[0]) + ',' + str(res[1])  + ',' + str(res[2]) + ',' + str(res[3]) + ',' + str(r2_res)
            print(line)
            plt.tight_layout()
            plt.show()
            # plt.savefig(os.path.join(SaveFile_Path_sum_base ,'Picture1',str(int(ids[i])) + '.png'))
            plt.close()

        line = str(ids[i]) + ',' + str(a1) + ',' + str(b1) + ',' + str(r2_1) + ',' +str(tau) + ',' + str(p_value) + ',' + str(res[0]) + ',' + str(res[1])  + ',' + str(res[2]) + ',' + str(res[3]) + ',' + str(r2_res)
        lines.append(line.replace('[','').replace(']',''))
    # writefile(os.path.join(SaveFile_Path_sum_base, 'bufferSumslopsen.csv'),lines)


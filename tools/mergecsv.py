from os import close
import pandas as pd
import os
import tqdm
import re
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

Folder_Path = r'C:\Users\hp\Desktop\wdpashp\wdpadata1\wdpa_ntl7'          #要拼接的文件夹及其完整路径，注意不要包含中文
SaveFile_Path =  r'C:\Users\hp\Desktop\wdpashp\wdpadata1'       #拼接后要保存的文件路径
# SaveFile_Name = r'all.csv'              #合并后要保存的文件名
 
# #修改当前工作目录
# os.chdir(Folder_Path)
# #将该文件夹下的所有文件名存入一个列表
# file_list = os.listdir()
 
# #读取第一个CSV文件并包含表头
# df = pd.read_csv(Folder_Path +'\\'+ file_list[0])   #编码默认UTF-8，若乱码自行更改
 
# #将读取的第一个CSV文件写入合并后的文件保存
# df.to_csv(SaveFile_Path+'\\'+ SaveFile_Name,encoding="utf_8_sig",index=False)
 
# #循环遍历列表中各个CSV文件名，并追加到合并后的文件
# for i in range(1,len(file_list)):
#     df = pd.read_csv(Folder_Path + '\\'+ file_list[i])
#     df.to_csv(SaveFile_Path+'\\'+ SaveFile_Name,encoding="utf_8_sig",index=False, header=False, mode='a+')

# basedf = None
# for i in tqdm.tqdm(range(1,51,1)):
#     df = pd.read_csv(os.path.join(Folder_Path,'WDPA_ntl_per_buffer'+str(i)+'.csv'))
#     if basedf is None:
#         df = df.drop(columns=['system:index','.geo'])
#         # df = df.rename(columns={'mean_':'mean'+str(i),'p0':'p0_'+str(i),
#         # 'p5':'p5_'+str(i),'p25':'p25_'+str(i),'p50':'p50_'+str(i),'p75':'p75_'+str(i),
#         # 'p95':'p95_'+str(i),'p100':'p100_'+str(i)})
#         basedf = df
#     else:
#         df = df.drop(columns=['system:index','area','iucn_cat','.geo'])
#         basedf = basedf.merge(df, on='objectid',how='left',suffixes = ('','_'+str(i)))
#         # .fillna("")

# basedf.to_csv(os.path.join(SaveFile_Path,"wdpa_merged.csv"), index=False)


# basedf = pd.read_csv(os.path.join(Folder_Path,'wdpa_ntl7_year.csv'))
# basedf = basedf.drop(columns=['system:index','.geo'])
# for i in tqdm.tqdm(range(1,51,1)):
#     df = pd.read_csv(os.path.join(Folder_Path,'wdpa_ntl7_year_buffer'+str(i)+'.csv'))
#     df = df.drop(columns=['system:index','area','iucn_cat','.geo'])
#     basedf = basedf.merge(df, on='objectid',how='left',suffixes = ('','_b'+str(i)))
#     # .fillna("")
# basedf.to_csv(os.path.join(SaveFile_Path,"wdpa_ntl7_merged.csv"), index=False)


####################################################################
pattern = re.compile('^dmsp2010_b(\d)+')
df= pd.read_csv(os.path.join(SaveFile_Path,"wdpa_ntl7_merged.csv"))

ntlID = np.array(df['objectid']).ravel()
cols = [i for i in df.columns if pattern.match(i)]
ntlValue = np.array(df[cols])



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

lines = []
for i in tqdm.tqdm(range(len(ntlID))):
# for i in range(len(ntlID)):
    if np.isnan(np.min(ntlValue[i,:])):
        line = str(ntlID[i])
    else:
        x = np.arange(1, 51, 1)
        y = ntlValue[i,:]
        max_index = np.argmax(y)
        ystr = ','.join([str(i) for i in y.tolist()])   

        if max_index == 0 or max_index == 50 or max_index == 1 or max_index == 49:
            y1,a1,b1,r2 = LinearModel(x.reshape(-1, 1),y.reshape(-1, 1))
            # plt.plot(x, y, 'k.')
            # plt.plot(x.reshape(-1, 1), y1, 'g-')
            # plt.title("max_index:"+str(max_index),size = 15)
            # # plt.show()
            # plt.savefig(os.path.join(SaveFile_Path ,'wdpa_ntl_picture2010\problem',str(ntlID[i]) + '.png'))
            # plt.close()
            line = str(ntlID[i]) + ',' + ystr + ',' + str(max_index) + ',' + str(a1) + ',' + str(b1) + ',' + str(0) + ',' + str(0) + ',' + str(r2) + ',' + str(0)

        else:
            y1,a1,b1,r2_1 = LinearModel(x[:max_index].reshape(-1, 1),y[:max_index].reshape(-1, 1))
            y2,a2,b2,r2_2 = LinearModel(x[max_index:].reshape(-1, 1),y[max_index:].reshape(-1, 1))
            line = str(ntlID[i]) + ',' + ystr + ',' + str(max_index) + ',' + str(a1) + ',' + str(b1) + ',' + str(a2) + ',' + str(a2) + ',' + str(r2_1) + ',' + str(r2_2)
            # plt.plot(x, y, 'k.')
            # plt.plot(x[:max_index].reshape(-1, 1), y1, 'g-')
            # plt.plot(x[max_index:].reshape(-1, 1), y2, 'b-')
            # plt.title("max_index:"+str(max_index),size = 15)
            # # plt.show()
            # # plt.savefig(os.path.join(SaveFile_Path ,'wdpa_ntl_picture2010',str(ntlID[i]) + '.png'))
            # plt.close()
        
    lines.append(line.replace('[','').replace(']',''))
writefile(os.path.join(SaveFile_Path, 'wdpa_ntl7_2010_maxValue.csv'),lines)           


# ###################################################################################
# basedf = None
# for i in tqdm.tqdm(range(25,51,1)):
#     df = pd.read_csv(os.path.join(Folder_Path,'wdpa_ntl_year_buffer'+str(i)+'.csv'))
#     if basedf is None:
#         df = df.drop(columns=['system:index','.geo'])
#         basedf = df
#     else:
#         df = df.drop(columns=['system:index','area','iucn_cat','.geo'])
#         basedf = basedf.merge(df, on='objectid',how='left',suffixes = ('','_b'+str(i)))
#         # .fillna("")

# basedf.to_csv(os.path.join(SaveFile_Path,"wdpa_ntl_merged_25after.csv"), index=False)
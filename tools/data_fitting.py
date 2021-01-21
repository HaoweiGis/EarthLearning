import os,sys
import argparse
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn import metrics

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert shp to semantic segmentation datasets')
    # parser.add_argument('--input', default=r'PKXNTL2/NTL_pkx_clean.csv' ,help='raster data path' )
    parser.add_argument('--input', default=r'C:\Users\hp\Desktop\贫困县\Analysis1207\pkx_pop_analysis.csv' ,help='raster data path' )
    # parser.add_argument('--testsize', default=0.3 ,help='raster data path' )
    # parser.add_argument('--batchsize', default=20 ,help='raster data path' )
    # parser.add_argument('--epoch', default=50 ,help='raster data path' )
    parser.add_argument('--output', default=r'C:\Users\hp\Desktop\贫困县\Analysis1207', help='output path')
    args = parser.parse_args()
    return args


def writefile(filename,lines):
    f = open(filename,'a')
    for line in lines:
        f.writelines(line + '\n')

if __name__ == "__main__":

    args = parse_args()
    
    df= pd.read_csv(args.input) 
    ntlValue = np.array(df.iloc[:,1:])
    ids = np.array(df.iloc[:,[0]]).ravel()

    def func(x,a,b,c,d):
        return a + b*(1/(1 + np.exp(-c*(x-d))))
    
    lines = []
    for i in range(len(ids)):
        x = np.arange(1, 22, 1)
        x_str = [str(i) for i in np.arange(2000, 2021, 1)]
        y = ntlValue[i,:]

        y_mean = np.mean(y)
        y_std = np.std(y)
        max = y_mean + (3*y_std)
        min = y_mean - (3*y_std)
        # max_zf = max - min + (2*y_std)
        # min_zf = max - min - y_std
        # print(max-min)
        # print(y_std)

        param_bounds=([min,0.1,0.8,1],[max,1,1,9])
        popt, pcov = curve_fit(func, x, y, bounds = param_bounds)
        a=popt[0] # popt里面是拟合系数，读者可以自己help其用法
        b=popt[1]
        c=popt[2]
        d=popt[3]
        print(a,b,c,d)
        
        yvals=func(x,a,b,c,d)
        # y_err = x.std() * np.sqrt(1/len(x)+
        # (x-x.mean())**2/np.sum((x-x.mean())**2))

        d_int = int(np.rint(d))
        # if d_int == 9:
        #     d_int = 8
        
        before_pk = np.mean(y[:d_int-1])
        after_pk = np.mean(y[d_int-1:])
        
        score = metrics.r2_score(y,yvals)
        mean_squared_error = metrics.mean_squared_error(y,yvals)
        ystr = ','.join([str(i) for i in y.tolist()])
        line = str(ids[i]) + ',' + ystr + ',' + str(a) + ',' + str(b) + ',' + str(c) + ',' + str(d) + ',' + str(d_int) + ',' + str(score) + ',' + str(mean_squared_error) + ',' + str(before_pk) + ',' + str(after_pk)  + ',' + str(d_int + 2012 -1)  + ',' + str(after_pk-before_pk)
        lines.append(line)

        csfont = {'family':'Times New Roman'}
        ax = plt.cla()
        plt.rc('font', family = 'Times New Roman')
        plot1=plt.scatter(x, y, label='Original Values')
        x_new = np.arange(1, 10, 0.2)
        yvals_new=func(x_new,a,b,c,d)
        plot2=plt.plot(x_new, yvals_new, c = 'r',label='Curve Fit Values')
        # plt.fill_between(x,yvals-y_err,yvals+y_err,alpha=0.2)
        plt.xlabel('Year',csfont,size = 12)
        plt.ylabel('Light Value',csfont, size = 12)
        plt.yticks( size = 10)
        plt.xticks(x, x_str, size = 10) 
        # labels = ax.get_xticklabels() + ax.get_yticklabels()
        # [label.set_fontname('Times New Roman') for label in labels]
        plt.legend(loc=4) # 指定legend的位置,读者可以自己help它的用法
        plt.title(str(ids[i]),csfont,size = 15)
        plt.savefig(os.path.join(args.output ,'Picture',str(ids[i]) + '.png'))
        # plt.show()
    writefile(os.path.join(args.output, 'Pop_calc_new.csv'),lines)
    
import os,sys
import argparse
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from scipy import optimize
from scipy.optimize import curve_fit
from sklearn import metrics
import lmfit
import tqdm

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert shp to semantic segmentation datasets')
    parser.add_argument('--input', default=r'C:\Users\hp\Desktop\wdpashp\wdpa_select100.csv' ,help='raster data path' )
    # parser.add_argument('--testsize', default=0.3 ,help='raster data path' )
    # parser.add_argument('--batchsize', default=20 ,help='raster data path' )
    # parser.add_argument('--epoch', default=50 ,help='raster data path' )
    parser.add_argument('--output', default=r'C:\Users\hp\Desktop\wdpashp\wdpadata', help='output path')
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

    def piecewise_linear(x, x0, y0, k1, k2):
        return np.piecewise(x, [x < x0, x >= x0], [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0])

    def err(w):
            th0 = w['th0'].value
            th1 = w['th1'].value
            th2 = w['th2'].value
            gamma = w['gamma'].value
            fit = th0 + th1*x + th2*np.maximum(0,x-gamma)
            return fit-y

    lines = []
    for i in tqdm.tqdm(range(len(ids))):
        if np.isnan(np.min(ntlValue[i,:])):
            pass
        else:
            x = np.arange(1, 51, 1)
            # x_str = [str(i) for i in np.arange(2012, 2021, 1)]
            y = ntlValue[i,:]

            p = lmfit.Parameters()
            p.add_many(('th0', 0.), ('th1', 0.0),('th2', 0.0),('gamma', 25.))
            mi = lmfit.minimize(err, p)
            b0 = mi.params['th0']; b1=mi.params['th1'];b2=mi.params['th2']
            gamma = int(mi.params['gamma'].value)
            
            X0 = np.array(range(1,gamma+1,1))
            X1 = np.array(range(0,51-gamma,1))
            y0 = y[:gamma-1+1]
            y1 = y[gamma-1:]
            Y0 = b0 + b1*X0
            Y1 = (b0 + b1 * float(gamma) + (b1 + b2)* X1)
            X = np.append(X0,X1+gamma)
            Y = np.append(Y0,Y1)
            # if gamma > self.start:
            # print(gamma)
            # print('value',y)
            # xz1 = int(gamma-((gamma-1)/2)-1)
            # xz2 = int(gamma+((21-gamma)/2)-1)
            # # print(xz1,xz2,gamma)
            # R_square0 = np.subtract(1,np.divide(np.sum(np.power(np.subtract(y0,Y0),2)),\
            # np.sum(np.power(np.subtract(y0,np.average(y0)),2))))
            # R_square1 = np.subtract(1,np.divide(np.sum(np.power(np.subtract(y1,Y1),2)),\
            # np.sum(np.power(np.subtract(y1,np.average(y1)),2))))
            # # print(R_square0,R_square1)
            # # print(X,Y)
            # plt.scatter(x, y,s =8)
            # plt.plot(X0[-1],Y0[-1],'ks')
            # plt.vlines(X0[-1],np.min(y),Y0[-1], colors = "c", linestyles = "dashed")
            # plt.vlines(X[xz1],np.min(y),Y[xz1], colors = "0.5", linestyles = "dashed")
            # plt.vlines(X[xz2],np.min(y),Y[xz2], colors = "0.5", linestyles = "dashed")
            # plt.ylim( np.min(y), )
            # plt.plot(X,Y,color='r')
            # plt.plot(X0,Y0,color='b',label = r'$y0=%s + %sx,R^2 =%s$' %(np.around(b0,2),np.around(b1,2),np.around(R_square0,decimals=2)))
            # plt.plot(X1+gamma,Y1,color='r',label = r'$y1=%s+%sx+%s(x-%s),R^2 =%s$'%(np.around(b0,2),np.around(b1,2),np.around(b2,2),gamma,np.around(R_square1,decimals=2)))
            # plt.xlabel('buffer')
            # plt.ylabel('Radiance Value'+'(nW '+r'$\mathbf{\mathrm{cm^{-2}}}$ '+ r'$sr^{-1}$)')
            # font1 = {'family' : 'Times New Roman',  
            #     'weight' : 'normal',  
            #     'size'   : 10,  
            #     }   
            # legend = plt.legend(prop=font1)#loc='upper right',
            # frame = legend.get_frame() 
            # frame.set_alpha(1) 
            # frame.set_facecolor('none')
            # plt.rc('font',family='Times New Roman')
            # pathfig = os.path.join(args.output ,'Picture50',str(ids[i]) + '.png')
            # plt.savefig(pathfig)
            # plt.close()
            # # plt.show()
            ystr = ','.join([str(i) for i in y.tolist()])
            line = str(ids[i]) + ',' + ystr + ',' + str(b0.value) + ',' + str(b1.value) + ',' + str(b2.value) + ',' + str(gamma)
            lines.append(line)

        # p , e = optimize.curve_fit(piecewise_linear, x, y)
        # xd = np.arange(1, 21, 0.2)
        # plt.plot(x, y, "o")
        # plt.plot(xd, piecewise_linear(xd, *p))
        # plt.show()

        # break
    #     y_mean = np.mean(y)
    #     y_std = np.std(y)
    #     max = y_mean + (3*y_std)
    #     min = y_mean - (3*y_std)
    #     # max_zf = max - min + (2*y_std)
    #     # min_zf = max - min - y_std
    #     # print(max-min)
    #     # print(y_std)

    #     param_bounds=([min,0.1,0.8,1],[max,1,1,9])
    #     popt, pcov = curve_fit(func, x, y, bounds = param_bounds)
    #     a=popt[0] # popt里面是拟合系数，读者可以自己help其用法
    #     b=popt[1]
    #     c=popt[2]
    #     d=popt[3]
    #     print(a,b,c,d)
        
    #     yvals=func(x,a,b,c,d)
    #     # y_err = x.std() * np.sqrt(1/len(x)+
    #     # (x-x.mean())**2/np.sum((x-x.mean())**2))

    #     d_int = int(np.rint(d))
    #     # if d_int == 9:
    #     #     d_int = 8
        
    #     before_pk = np.mean(y[:d_int-1])
    #     after_pk = np.mean(y[d_int-1:])
        
    #     score = metrics.r2_score(y,yvals)
    #     mean_squared_error = metrics.mean_squared_error(y,yvals)
    #     ystr = ','.join([str(i) for i in y.tolist()])
    #     line = str(ids[i]) + ',' + ystr + ',' + str(a) + ',' + str(b) + ',' + str(c) + ',' + str(d) + ',' + str(d_int) + ',' + str(score) + ',' + str(mean_squared_error) + ',' + str(before_pk) + ',' + str(after_pk)  + ',' + str(d_int + 2012 -1)  + ',' + str(after_pk-before_pk)
    #     lines.append(line)

    #     csfont = {'family':'Times New Roman'}
    #     ax = plt.cla()
    #     plt.rc('font', family = 'Times New Roman')
    #     plot1=plt.scatter(x, y, label='Original Values')
    #     x_new = np.arange(1, 21, 0.2)
    #     yvals_new=func(x_new,a,b,c,d)
    #     plot2=plt.plot(x_new, yvals_new, c = 'r',label='Curve Fit Values')
    #     # plt.fill_between(x,yvals-y_err,yvals+y_err,alpha=0.2)
    #     plt.xlabel('Year',csfont,size = 12)
    #     plt.ylabel('Light Value',csfont, size = 12)
    #     plt.yticks( size = 10)
    #     # plt.xticks(x, x_str, size = 10) 
    #     # labels = ax.get_xticklabels() + ax.get_yticklabels()
    #     # [label.set_fontname('Times New Roman') for label in labels]
    #     plt.legend(loc=4) # 指定legend的位置,读者可以自己help它的用法
    #     plt.title(str(ids[i]),csfont,size = 15)
    #     # plt.savefig(os.path.join(args.output ,'Picture',str(ids[i]) + '.png'))
    #     plt.show()
    writefile(os.path.join(args.output, 'NLT_calc_np.csv'),lines)
    
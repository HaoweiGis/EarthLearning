from tkinter.constants import N
from unittest import result
import numpy as np
import argparse
from osgeo import gdal,ogr
# import matplotlib.pyplot  as plt
import os,sys
import subprocess

from numpy.core.fromnumeric import shape
from skimage import measure
from tqdm import tqdm

import geotools

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert shp to semantic segmentation datasets')
    parser.add_argument('--input', default=r'C:\Users\hp\Desktop\Ecologicalproject\HabitatFragmentation\ndvi2020.tif' ,help='raster data path' )
    parser.add_argument('--type', default='classification' )
    parser.add_argument('--output', default=r'C:\Users\hp\Desktop\Ecologicalproject\HabitatFragmentation\Habitat', help='output path')
    args = parser.parse_args()
    return args

def pad(X, margin):
    newX = np.zeros((X.shape[0], X.shape[1]+margin*2, X.shape[2]+margin*2))
    newX[:, margin:X.shape[1]+margin, margin:X.shape[2]+margin] = X
    return newX

def patch(X, patch_size, height_index, width_index):
    height_slice = slice(height_index, height_index+patch_size)
    width_slice = slice(width_index, width_index+patch_size)
    patch = X[:, height_slice, width_slice]
    return np.nan_to_num(patch)


def BatchCreate(im_data, im_porj, im_geotrans):
    im_mask = None
    im_original = None
    for status in tqdm(range(1,101,1)):
        ntl_mask=np.zeros(im_data.shape)
        ntl_original = np.zeros(im_data.shape)
        indexv = np.where(im_data>=status)
        
        ntl_original[indexv]= im_data[indexv]
        ntl_mask[indexv]= 1

        # 输出保存成单波段
        geotools.GeoImgW(os.path.join(args.output, 'original/status'+ str(status)+'.tif'), ntl_original, im_geotrans, im_porj)    
        geotools.GeoImgW(os.path.join(args.output, 'mask/status'+ str(status)+'.tif'), ntl_mask, im_geotrans, im_porj)

        # 输出保存成多波段
        # ntl_class = label(ntl_class, connectivity = 1)
        # if im_mask is None:
        #     im_mask = ntl_mask
        #     im_original = ntl_original
        # else:   
        #     im_mask = np.concatenate((im_mask, ntl_mask), axis=0)     
        #     im_original = np.concatenate((im_original,ntl_original),axis = 0)
    # return im_mask,im_original

def StatusData(self):
    # self.CalcPerimeter()
    self.CalcIndex()
    PatchSum,PatchType,LPISum,AreaRate,PdSum,MpsSum,SHDISum,ViSum,SHEISum,SiSum = self.CalcIndex()
    # print(np.array(PatchType).shape,np.array(LPISum).shape,np.array(AreaRate).shape)
    # print(PatchSum,LPISum,AreaRate)
    self.BatchPlot(PatchSum,self.max_value,'Sum plaques')
    self.BatchPlot(LPISum,self.max_value-1,'LPISum')
    self.BatchPlot(AreaRate,self.max_value-1,'AreaRate')
    self.BatchPlot(PdSum,self.max_value-1,'PdSum')
    self.BatchPlot(MpsSum,self.max_value-1,'MpsSum')
    self.BatchPlot(SHEISum,self.max_value-1,'SHEISum')
    self.BatchPlot(SHDISum,self.max_value-1,'SHDISum')
    self.BatchPlot(ViSum,self.max_value-1,'ViSum')
    self.BatchPlot(SiSum,self.max_value-1,'SiSum')
    Died = []
    Unchange = []
    Shrink = []
    Split = []
    for i in PatchType:
        # print(i)
        Slist = i.split(',') 
        Died.append(int(Slist[0]))
        Unchange.append(int(Slist[1]))
        Shrink.append(int(Slist[2]))
        Split.append(int(Slist[3]))
    # print(Died,'\n',Unchange,'\n',Shrink,'\n',Split)
    self.BatchPlot(Died,self.max_value-1,'Dead plaque')
    self.BatchPlot(Unchange,self.max_value-1,'Unchange plaques')
    self.BatchPlot(Shrink,self.max_value-1,'Shrink plaques')
    self.BatchPlot(Split,self.max_value-1,'Split plaques')

def FileWrite(lines):
    with open(args.output + '/PatchSum.csv','a') as f:
        for line in lines:
            f.writelines(line)


if __name__ == "__main__":
    args = parse_args()

    # 1批量生成分层NDVI数据和Mask
    # im_data, im_porj, im_geotrans = geotools.GeoImgR(args.input)
    # im_shape = im_data.shape
    # im_data = im_data*100
    # #geotools.GeoImgW('im_data.tif', im_data, im_geotrans, im_porj)
    # BatchCreate(im_data, im_porj, im_geotrans)

    # # 2对filter之后的数据转矢量
    # for status in tqdm(range(1,101,1)):
    #     filtermask = os.path.join(args.output, 'maskfilter/status'+ str(status)+'.tif')
    #     shpfile = os.path.join(args.output, 'shpfile/status'+ str(status)+'.shp')
    #     geotools.RasterToVector(filtermask,shpfile)
    
    # # skimage 
    # lines = []
    # for status in tqdm(range(1,101,1)):
    #     filtermask = os.path.join(args.output, 'maskfilter/status'+ str(status)+'.tif')
    #     im_data, im_porj, im_geotrans = geotools.GeoImgR(filtermask)
    #     im_data = np.squeeze(im_data).astype(int)
    #     # im_data[np.where(im_data == 0)] = np.nan
    #     label_image =measure.label(im_data,background=0)
    #     NP = len(measure.regionprops(label_image))
    #     st_area = None
    #     for region in measure.regionprops(label_image):
    #         if st_area is None:
    #             st_area = region.area
    #         else:
    #             st_area = np.append(st_area,region.area)
    #     if st_area is None:
    #         line = str(status) + '\n'
    #     else:
    #         LPI = np.divide(st_area.max(),st_area.sum())
    #         MPS = st_area.mean()
    #         line = str(status) + ',' + str(NP) + ',' + str(LPI) + ',' + str(MPS) + '\n'
    #     lines.append(line)
    # FileWrite(lines)

    for status in tqdm(range(1,100,1)):
        PatchSum = []
        filetiff1 = os.path.join(args.output, 'maskfilter/status'+ str(status)+'.tif')
        filetiff2 = os.path.join(args.output, 'maskfilter/status'+ str(status+1)+'.tif')
        im_data1, im_porj1, im_geotrans1 = geotools.GeoImgR(filetiff1)
        im_data2, im_porj2, im_geotrans2 = geotools.GeoImgR(filetiff2)
        im_data = np.squeeze(im_data1).astype(int)
        label_image1 =measure.label(im_data1,background=0)
        label_image2 =measure.label(im_data2,background=0)
        num_p_1=len(np.unique(im_data1))
        num_p_2=len(np.unique(im_data2))
        PatchSum.append(num_p_1)
        if status == 100-1: PatchSum.append(num_p_2)
        # print(PatchSum)
        Died = 0
        Unchange = 0
        Shrink = 0
        Split = 0
        areasum1 = len(np.where(im_data1!=0)[0])
        areasum2 = len(np.where(im_data2!=0)[0])
        # print(arearate)
        arealist =[]
        for j in range(1,num_p_1):   #j plaque        
            index = np.where(im_data1==j)
            arealist.append(len(index[0])) 
            id_p2=np.unique(im_data2[index]) #代表2图像中的斑块数
            # print(id_p2)
            if len(id_p2)==1 and id_p2==0: #如果现在图像中只有一个值且为0，那么原斑块消失，所以state记录原图像状态
                Died = Died +1   #消亡 
            elif len(id_p2)==1 and id_p2 !=0:
                Unchange = Unchange + 1  #未变
            elif len(id_p2)==2 and sum(id_p2==0)==1: 
                Shrink = Shrink + 1   #收缩
            elif len(id_p2)>=2 :
                Split = Split + 1   #分裂        
        Line = str(Died)+','+str(Unchange)+','+str(Shrink)+','+str(Split)

    # filtermask = os.path.join(args.output, 'maskfilter/status'+ str(3)+'.tif')
    # im_data, im_porj, im_geotrans = geotools.GeoImgR(filtermask)
    # im_data = np.squeeze(im_data).astype(int)
    # # im_data[np.where(im_data == 0)] = np.nan
    # from skimage import measure,morphology,color
    # labels =measure.label(im_data,connectivity=1,background=0)
    # dst=color.label2rgb(labels)  #根据不同的标记显示不同的颜色
    # print('regions number:',len(measure.regionprops(labels)))  #显示连通区域块数(从0开始标记)
    # import matplotlib.pyplot as plt
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    # ax1.imshow(im_data, plt.cm.gray, interpolation='nearest')
    # ax1.axis('off')
    # ax2.imshow(dst,interpolation='nearest')
    # ax2.axis('off')

    # fig.tight_layout()
    # plt.show()
    # label_images = measure.regionprops(im_data)
    # print(len(label_images))


    # 化简
    # for status in tqdm(range(50,100,5)):
    #     inputtif = os.path.join(args.output, 'ntl_mask'+ str(status)+'.tif')
    #     outtif = os.path.join(args.output, 'maskfilter','ntl_mask'+ str(status)+'.tif')
    #     geotools.filterImg(inputtif,outtif)

    # im_mask,im_original = BatchCreate(im_data, im_porj, im_geotrans)
    # geotools.GeoImgW('im_mask.tif', im_mask, im_geotrans, im_porj)
    # geotools.GeoImgW('im_original.tif', im_original, im_geotrans, im_porj)

    # Rao Q diversity
    # outputs = np.zeros((1 ,im_shape[1],
    #                 (im_shape[2]))) 
    # slice_size = 3  # 移动视窗尺寸 与切片尺寸一致
    # im_pad = pad(im_data, 1)

    # for h in tqdm(range(1000)):
    #     for w in range(im_data.shape[2]):
    #         if np.isnan(im_data[:,h,w]):
    #             diversity = np.nan
    #         else:
    #             windows = patch(im_pad, slice_size, h, w).flatten()
    #             rao = None
    #             for p in windows:
    #                 if rao is None:
    #                     rao = 1/(1+np.abs(windows-p))/9
    #                 else:
    #                     rao = np.hstack(((1/(1+np.abs(windows-p)))/9))
    #             diversity = np.sum(rao)
    #         outputs[0 ,h, w] = diversity
    
    # geotools.GeoImgW(args.output,outputs,im_geotrans,im_porj)



    
    
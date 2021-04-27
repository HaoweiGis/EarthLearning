import random
import gdal
import numpy as np
import os
import argparse
from tqdm import tqdm
import glob

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert shp to semantic segmentation datasets')
    parser.add_argument('--inputDir', default=r'D:\2_HaoweiPapers\6_MappingForSlum\Datasets\Images' ,help='raster data path' )
    parser.add_argument('--outputDir', default=r'D:\2_HaoweiPapers\6_MappingForSlum\Datasets\deeps', help='output path')
    # parser.add_argument('--stride', default=450, help='support to byte')
    parser.add_argument('--RGB', default=None, help='para si list,example [3,2,1]')
    parser.add_argument('--clipsize', default=512, help='support to byte')
    args = parser.parse_args()
    return args

#  读取tif数据集
def readTif(fileName):
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName + "文件无法打开")
    return dataset
    
#  保存tif文件函数
def writeTiff(im_data, im_geotrans, im_proj, path):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])
        im_bands, im_height, im_width = im_data.shape
    #创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, int(im_width), int(im_height), int(im_bands), datatype)
    if(dataset!= None):
        dataset.SetGeoTransform(im_geotrans) #写入仿射变换参数
        dataset.SetProjection(im_proj) #写入投影
    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset
    
'''
随机裁剪函数
ImagePath 原始影像路径
LabelPath 标签影像路径
IamgeSavePath 原始影像裁剪后保存目录
LabelSavePath 标签影像裁剪后保存目录
CropSize 裁剪尺寸
CutNum 裁剪数量
'''
def RandomCrop(ImagePath, LabelPath, IamgeSavePath, LabelSavePath, CropSize, CutNum):
    dataset_img = readTif(ImagePath)
    width = dataset_img.RasterXSize
    height = dataset_img.RasterYSize
    proj = dataset_img.GetProjection()
    geotrans = dataset_img.GetGeoTransform()
    img = dataset_img.ReadAsArray(0,0,width,height)#获取哟昂数据
    dataset_label = readTif(LabelPath)
    label = dataset_label.ReadAsArray(0,0,width,height)#获取标签数据
    
    #  获取当前文件夹的文件个数len,并以len+1命名即将裁剪得到的图像
    fileNum = len(os.listdir(IamgeSavePath))
    new_name = fileNum + 1
    while(new_name < CutNum + fileNum + 1):
        #  生成剪切图像的左上角XY坐标
        UpperLeftX = random.randint(0, height - CropSize)    
        UpperLeftY = random.randint(0, width - CropSize)    
        if(len(img.shape) == 2):
            imgCrop = img[UpperLeftX : UpperLeftX + CropSize,
                          UpperLeftY : UpperLeftY + CropSize]
        else:
            imgCrop = img[:,
                          UpperLeftX : UpperLeftX + CropSize,
                          UpperLeftY : UpperLeftY + CropSize]
        if(len(label.shape) == 2):
            labelCrop = label[UpperLeftX : UpperLeftX + CropSize,
                              UpperLeftY : UpperLeftY + CropSize]
        else:
            labelCrop = label[:,
                              UpperLeftX : UpperLeftX + CropSize,
                              UpperLeftY : UpperLeftY + CropSize]
        writeTiff(imgCrop, geotrans, proj, IamgeSavePath + "/%d.tif"%new_name)
        writeTiff(labelCrop, geotrans, proj, LabelSavePath + "/%d.tif"%new_name)
        new_name = new_name + 1
        
# #  裁剪得到300张256*256大小的训练集         
# RandomCrop(r"Data\data2\tif\data2.tif",
#            r"Data\data2\label\label.tif",
#            r"Data\train\image1",
#            r"Data\train\label1",
#            256,300)

if __name__ == "__main__":
    args = parse_args()
    inputImgs = glob.glob(args.inputDir + '/*.tif')

    for img in tqdm(inputImgs):
        geoimg = gdal.Open(img)
        im_width = geoimg.RasterXSize
        im_height = geoimg.RasterYSize
        if im_width> 512 and im_height>512:
            labelImg = img.replace('Images', 'labels').replace('.tif','_label.tif')
            RandomCrop(img,
            labelImg,
            args.outputDir + '/images',
            args.outputDir + '/labels',
            512,2)

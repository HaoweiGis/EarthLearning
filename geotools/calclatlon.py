import numpy as np
import argparse
import os.path as osp
import geotools

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert shp to semantic segmentation datasets')
    parser.add_argument('--input', default=r'‪LULC\LULC1000\lulc1980.tif' ,help='raster data path' )
    parser.add_argument('--type', default='classification' )
    parser.add_argument('--output', default=r'‪LULC\Excel', help='output path')
    args = parser.parse_args()
    return args

def writefile(filename,lines):
    f = open(filename,'a')
    for line in lines:
        linestr = str(line)
        f.writelines(linestr + '\n')

if __name__ == "__main__":
    args = parse_args()
    args.input = args.input.strip('\u202a')
    args.output = args.output.strip('\u202a')
    im_data, im_porj, im_geotrans = geotools.GeoImgR(args.input)
    # imgarr = im_data
    # 沿着经纬度进行数据统计mean
    # imgarr = im_data[3,:,:].reshape(im_data.shape[1],im_data.shape[2])
    # imgarr[np.where(np.abs(imgarr)>99999)] = np.nan
    # # geotools.GeoImgW('rsei1.tif', imgarr, im_geotrans, im_porj)
    # lon = np.nanmean(imgarr,axis=1)
    # lat = np.nanmean(imgarr,axis=0)
    # writefile(osp.join(args.output,'latndsi.csv'),lat)
    # writefile(osp.join(args.output,'lonndsi.csv'),lon)
    # # print(lon, lat)

    # 沿着经纬度进行数据统计面积
    imgarr = im_data.reshape(im_data.shape[1],im_data.shape[2])
    # imgarr[np.where((imgarr==31)|(imgarr==32)|(imgarr==33))] = 1
    imgarr[np.where((imgarr==21)|(imgarr==22)|(imgarr==23)|(imgarr==24))] = 1
    imgarr[np.where(imgarr!=1)] = 0
    lon = np.sum(imgarr,axis=1)
    lat = np.sum(imgarr,axis=0)
    writefile(osp.join(args.output,'lat1980.csv'),lat)
    writefile(osp.join(args.output,'lon1980.csv'),lon)
    # print(lon, lat)

    # imgarr = im_data.reshape(im_data.shape[1],im_data.shape[2])
    # bin = [11,13,26,34,44,45,47,54,63,70]
    # histogramImg = np.histogram(imgarr,bin)
    # bin = [0,13,26,34,44,45,47,54,63,70]
    # histogramImg = np.histogram(imgarr,bin)
    # print(histogramImg)

    
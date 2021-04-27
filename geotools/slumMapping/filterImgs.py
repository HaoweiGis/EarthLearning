import os
import argparse
import gdal
from tqdm import tqdm
import glob
import numpy as np

import sys
sys.path.append(r'C:\Users\dell\Documents\GitHub\EarthLearning\geotools')
# print(sys.path)
import gdal_base

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert shp to semantic segmentation datasets')
    parser.add_argument('--inputDir', default=r'D:\2_HaoweiPapers\6_MappingForSlum\Datasets\deeps' ,help='raster data path' )
    parser.add_argument('--outputDir', default=r'D:\2_HaoweiPapers\6_MappingForSlum\Datasets\slumSamples', help='output path')
    args = parser.parse_args()
    return args


def FileWrite(filename,lines):
    with open(filename,'a') as f:
        for line in lines:
            f.writelines(line)


if __name__ == "__main__":
    args = parse_args()
    
    lines = []
    for num in tqdm(range(1,6524,1)):
        img = os.path.join(args.inputDir + '/images',str(num)+'.tif')
        label = os.path.join(args.inputDir + '/labels',str(num)+'.tif')

        im_data, im_porj, im_geotrans = gdal_base.GeoImgR(img)
        lab_data, lab_porj, lab_geotrans = gdal_base.GeoImgR(label)

        con1 = np.where(np.sum(im_data, axis=0) == 765)[0].size
        tureNum = np.where(lab_data == 1)[0].size 
        con2 = tureNum/262144
        if con1 < 1000 and con2 > 0.15:
            lines.append(str(num)+'.tif' + '\n')
        # print(num)
    filename = args.outputDir + '/all.txt'
    FileWrite(filename,lines)
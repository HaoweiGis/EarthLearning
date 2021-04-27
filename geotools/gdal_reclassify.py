import os
import argparse
import gdal
# import module.utils as geotools
import gdal_base
from tqdm import tqdm
import glob
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert shp to semantic segmentation datasets')
    parser.add_argument('--inputDir', default=r'D:\2_HaoweiPapers\6_MappingForSlum\Datasets\label' ,help='raster data path' )
    parser.add_argument('--outputDir', default=r'D:\2_HaoweiPapers\6_MappingForSlum\Datasets\labels', help='output path')
    args = parser.parse_args()
    return args




if __name__ == "__main__":
    args = parse_args()
    inputImgs = glob.glob(args.inputDir + '/*.tif')
    
    for img in inputImgs:
        filename = os.path.basename(img)
        im_data, im_porj, im_geotrans = gdal_base.GeoImgR(img)
        # im_data[np.where(im_data == 0)] = 2
        im_data[np.where(im_data == 0)] = np.nan
        gdal_base.GeoImgW(args.outputDir + '/' + filename,im_data, im_geotrans, im_porj,driver='GTiff')
        print(im_data)
        
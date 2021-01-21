'''
@File    :   img2point.py
@Time    :   2020/08/30 19:40:08
@Author  :   Haowei
@Version :   1.0
@Contact :   blackmhw@gmail.com
@Desc    :   None
'''


from osgeo import gdal,ogr
import argparse
import numpy as np
import module.utils as geotool

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert shp to semantic segmentation datasets')
    parser.add_argument('--shp', default=r'D:\EarthLearning\preprocessing\datasets\landcover\piont\Sampleshp3857.shp' ,help='raster data path' )
    parser.add_argument('--image', default=r'D:\EarthLearning\preprocessing\datasets\landcover\piont\SampleS2.tif' ,help='raster data path' )
    parser.add_argument('--output', default=r'D:\EarthLearning\preprocessing\datasets\landcover\piont', help='output path')
    parser.add_argument('--stride', default=200, help='support to byte')
    parser.add_argument('--RGB', default=None, help='para is list,example [3,2,1]')
    parser.add_argument('--clipsize', default=512, help='support to byte')
    parser.add_argument('--classtype', default='1', help='support to byte')
    args = parser.parse_args()
    return args

def writefile(filename,lines):
    f = open(filename,'a')
    for line in lines:
        f.writelines(line + '\n')

if __name__ == "__main__":
    args = parse_args()

    src_filename = args.image
    shp_filename = args.shp

    lines = geotool.img2point(src_filename,shp_filename)
    writefile('sample.csv',lines)
        

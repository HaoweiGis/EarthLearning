'''
@File    :   img2tile.py
@Time    :   2020/08/28 20:15:20
@Author  :   Haowei
@Version :   1.0
@Contact :   blackmhw@gmail.com
@Desc    :   None
'''
import os
import argparse
import gdal
# import module.utils as geotools
import geotools
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert shp to semantic segmentation datasets')
    parser.add_argument('--image', default=r'D:\EarthLearning\preprocessing\datasets\landcover\tile\landcover3_img10.tif' ,help='raster data path' )
    parser.add_argument('--output', default=r'D:\EarthLearning\preprocessing\datasets\landcover\tile\lable', help='output path')
    parser.add_argument('--stride', default=450, help='support to byte')
    parser.add_argument('--RGB', default=None, help='para si list,example [3,2,1]')
    parser.add_argument('--clipsize', default=512, help='support to byte')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    geoimg = gdal.Open(args.image)
    im_width = geoimg.RasterXSize
    im_height = geoimg.RasterYSize
    tile_name = len(os.listdir(args.output)) + 1
    start_num = tile_name
    for y in tqdm(range(0, im_height-args.stride, args.stride)):
        for x in range(0, im_width-args.stride, args.stride):
            out_file = os.path.join(args.output,str(tile_name)+".tif")
            geotools.img2tile(args.image, out_file, x, y, args.clipsize,RGB=args.RGB)
            tile_name = tile_name + 1
    print('geotool process tile is: ',tile_name-start_num)
    # geotool.img2tile(args.image, 'gouzi.tif', 3000, 0, 512)

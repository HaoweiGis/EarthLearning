import argparse

import geotools


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert shp to semantic segmentation datasets')
    parser.add_argument('--input', default=r'C:\Users\hp\Desktop\Ecologicalproject\HabitatFragmentation\ndvi2020.tif' ,help='raster data path' )
    parser.add_argument('--type', default='classification' )
    parser.add_argument('--output', default=r'C:\Users\hp\Desktop\Ecologicalproject\HabitatFragmentation\ndvi2020diver.tif', help='output path')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    intimg = r'C:\Users\hp\Desktop\Ecologicalproject\HabitatFragmentation\GLC2020\beijing3857.tif'
    targetimg = r'C:\Users\hp\Desktop\Ecologicalproject\HabitatFragmentation\ndvi2020nodata.tif'
    outimg = r'C:\Users\hp\Desktop\Ecologicalproject\HabitatFragmentation\beijing_lulc.tif'
    geotools.reSamplebyimg(intimg,targetimg,outimg,pixsize=30)
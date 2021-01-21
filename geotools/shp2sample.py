'''
@File    :   main.py
@Time    :   2020/08/28 11:01:52
@Author  :   Haowei
@Version :   1.0
@Contact :   blackmhw@gmail.com
@Desc    :   
intput：样本shp目录 (shp文件必须包含class类别,EPSG:3857);
        多光谱的影像数据 (EPSG:3857)
output：在输出目录中生成包含label和对应区域的image影像
'''
import argparse
import os.path as osp
import glob
import sys

# sys.path.append('./') 
# import module.utils as geotool
import geotools as geotool

def collect_files(dir,suffix):
    files = []
    filen = osp.join(dir, '*.' + suffix)
    filenames = glob.glob(filen)
    for filename in filenames:
        assert filename.endswith(suffix), filename
        files.append(filename)
    assert len(files), f'No images found in {dir}'
    print(f'Loaded {len(files)} images from {dir}')
    return files

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert shp to semantic segmentation datasets')
    parser.add_argument('--shpdir', default=r'D:\gaofen_caigangfang\Sample', help='shp data path')
    parser.add_argument('--image', default=r'D:\gaofen_caigangfang\Land_GF2.tif' ,help='raster data path' )
    parser.add_argument('--output', default=r'D:\gaofen_caigangfang\shpimg', help='output path')
    parser.add_argument('--classtype', default='fasdfs', help='support to byte')
    parser.add_argument('--datatype', default='byte', help='support to byte')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    # # shp to geoTiff
    shpfiles = collect_files(args.shpdir,'shp')
    for shpfile in shpfiles:
        # support muilt-class  (还需要修改)
        geotool.shp2img(shpfile,args.image,args.output,args.classtype)

    # # data type convert byte
    # if args.datatype == 'byte':
    #     geotool.img2byte(args.image,args.output)
    
    # # img clip img (employ gdal_translate is not resample,therefore crs is consistent)
    # for shpfile in shpfiles:
    #     shpname = osp.basename(shpfile).split('.')[0]+'_'+ args.classtype
    #     outtif = osp.join(args.output,shpname + '.tif')
    #     geotool.rasterClipraster(outtif,args.image,args.output)

    # "所有函数需要明确输入输出，保持标准化"

    # outtif = r'D:\EarthLearning\preprocessing\datasets\landcover\landcover1_byte.tif'
    # geotool.rasterClipraster(outtif,args.image,args.output)

    intimg = r'D:\EarthLearning\preprocessing\datasets\landcover\tile\mosaic_3857.tif'
    targetimg = r'D:\EarthLearning\preprocessing\datasets\landcover\tile\landcover3_byte.tif'
    outimg = r'D:\EarthLearning\preprocessing\datasets\landcover\tile\label30\landcover3_img10.tif'
    geotools.reSamplebyimg(intimg,targetimg,outimg)
        


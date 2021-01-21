'''
@File    :   geotools.py
@Time    :   2020/09/03 15:14:30
@Author  :   Haowei
@Version :   1.0
@Contact :   blackmhw@gmail.com
@Desc    :   None
'''

import os.path as osp
import os
import subprocess
import numpy as np
from numpy.core.fromnumeric import shape

from osgeo import gdal,ogr,osr
from gdalconst import GA_ReadOnly
from scipy.signal.filter_design import band_stop_obj

# commond line:     
# 拼接，裁剪
# https://gdal.org/programs/gdal_merge.html
# gdal_merge.py -init 255 -o out.tif in1.tif in2.tif
# https://gdal.org/programs/gdalwarp.html
# gdalwarp -q  -crop_to_cutline -cutline ../shpfile/beijing.shp -t_srs EPSG:3857 -tr 30 30 -of GTiff beijing.tif beijing_clip.tif


def GeoImgR(filename):
    dataset = gdal.Open(filename)
    im_porj = dataset.GetProjection()
    im_geotrans = dataset.GetGeoTransform()
    im_data = np.array(dataset.ReadAsArray())
    if len(im_data.shape) == 2:
        im_data = im_data[np.newaxis,:, :]
    del dataset
    return im_data, im_porj, im_geotrans

def GeoImgW(filename,im_data, im_geotrans, im_porj,driver='GTiff'):
    im_shape = im_data.shape
    driver = gdal.GetDriverByName(driver)
    if "int8" in im_data.dtype.name:
        datetype = gdal.GDT_Byte
    elif "int16" in im_data.dtype.name:
        datetype = gdal.GDT_UInt16
    elif "int32" in im_data.dtype.name:
        datetype = gdal.GDT_UInt32
    else :
        datetype = gdal.GDT_Float32
    # driver.Create weight hight
    dataset = driver.Create(filename, im_shape[2], im_shape[1], im_shape[0], datetype)
    dataset.SetGeoTransform(im_geotrans)
    dataset.SetProjection(im_porj)
    for band_num in range(im_shape[0]):
        img = im_data[band_num,:,:]
        band_num = band_num + 1
        dataset.GetRasterBand(band_num).WriteArray(img)
    del dataset

def img2point(src_filename,shp_filename):
    src_ds=gdal.Open(src_filename) 
    gt=src_ds.GetGeoTransform()

    ds=ogr.Open(shp_filename)
    lyr=ds.GetLayer()
    print('Sample feature number is: ',lyr.GetFeatureCount())
    lines = []
    for feat in lyr:
        geom = feat.GetGeometryRef()
        land_use = feat.GetField('land_use')
        mx,my=geom.GetX(), geom.GetY()  #coord in map units

        #Convert from map to pixel coordinates.
        #Only works for geotransforms with no rotation.
        px = int((mx - gt[0]) / gt[1]) #x pixel
        py = int((my - gt[3]) / gt[5]) #y pixel

        intval=src_ds.ReadAsArray(px,py,1,1)
        flateenarr = intval.flatten()[:-4]
        dataline = np.insert(flateenarr,0,land_use)
        dataline = [str(i) for i in dataline]
        line = ','.join(dataline)
        lines.append(line)
    return lines

def imgStats(geofile,band):
    '''
    Statistic image maximum and minimum values
    '''
    srcband = geofile.GetRasterBand(band)
    # Get raster statistics
    stats = srcband.GetStatistics(True, True)
    return stats[0],stats[1]

def img2byte(imgfile,output):
    '''
    open raster and choose band to find min, max
    '''
    imgname = osp.basename(imgfile).split('.')[0]+'_byte'
    outtif = osp.join(output,imgname + '.tif')
    print(outtif)
    geodataset = gdal.Open(imgfile)
    bands_num = geodataset.RasterCount
    minarr = []
    maxarr = []
    for band in range(1,bands_num+1):
        minI,maxI= imgStats(geodataset,band)
        minarr.append(minI)
        maxarr.append(maxI)
    minV = str(min(minarr))
    maxV = str(max(maxarr))
    cmd1 = ['gdal_translate', "-ot", "Byte", "-of","GTiff","-scale",minV,maxV,'0','255', imgfile, outtif]
    print(' '.join(cmd1))
    subprocess.call(cmd1)
    # os.system('gdal_translate -of GTiff -ot Byte -scale ' + ' '.join([str(x) for x in [minV, maxV, 0, 255]]) + ' -of GTiff '+ imgfile + ' ' + outtif)
    print(imgfile + ' img2byte is finally!'+'\n')

def filterImg(inputtif, outtif,threshold=1000):
    cmd = ['gdal_sieve.py', "-st", str(threshold), inputtif, outtif]
    print(' '.join(cmd))
    subprocess.call(cmd)

def RasterToVector(tiffile,shpfile,maskfile = None):
    dataset = gdal.Open(tiffile)
    porj = dataset.GetProjection()
    srcband = dataset.GetRasterBand(1)
    srcband.SetNoDataValue(0)
    driver = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(shpfile):
        driver.DeleteDataSource(shpfile)
    outDatasource = driver.CreateDataSource(shpfile)
    srs = osr.SpatialReference()
    srs.ImportFromWkt(porj)
    outLayer = outDatasource.CreateLayer("polygonized", srs = srs)
    oFieldID = ogr.FieldDefn('DN',ogr.OFTInteger)
    outLayer.CreateField(oFieldID, 1)
    gdal.Polygonize( srcband, srcband, outLayer,0 , [], callback=None )
    outDatasource.Destroy()


def shp2img(shpfile,targeImg,output,classtype={'class':'801001'}):
    '''
    shp transform raster and nead reslution
    '''
    shpname = osp.basename(shpfile).split('.')[0]+'_'+ str(1)
    outtif = osp.join(output,shpname + '.tif')
    
    dataimg = gdal.Open(targeImg)
    res = str(dataimg.GetGeoTransform()[1])
    (key, value), = classtype.items()
    cmd1 = ['gdal_rasterize', "-ot", "Byte", "-a",key,"-where",key +"='" + value +'\'',"-tr",res, res, "-init", "0", "-a_nodata", "0",shpfile,outtif]
    print(' '.join(cmd1))
    subprocess.call(cmd1)

    # data = gdal.Open(targeImg, gdalconst.GA_ReadOnly)
    # geo_transform = data.GetGeoTransform()
    # #source_layer = data.GetLayer()
    # x_min = geo_transform[0]
    # y_max = geo_transform[3]
    # x_max = x_min + geo_transform[1] * data.RasterXSize
    # y_min = y_max + geo_transform[5] * data.RasterYSize
    # x_res = data.RasterXSize
    # y_res = data.RasterYSize
    # mb_v = ogr.Open(shpfile)
    # mb_l = mb_v.GetLayer()
    # pixel_width = geo_transform[1]
    # target_ds = gdal.GetDriverByName('GTiff').Create(outtif, x_res, y_res, 1, gdal.GDT_Byte)
    # target_ds.SetGeoTransform((x_min, pixel_width, 0, y_min, 0, pixel_width))
    # band = target_ds.GetRasterBand(1)
    # NoData_value = 0
    # band.SetNoDataValue(NoData_value)
    # band.FlushCache()
    # gdal.RasterizeLayer(target_ds, [1], mb_l)
    # # gdal.RasterizeLayer(target_ds, [1], mb_l, options=["ATTRIBUTE=hedgerow"])
    # # target_ds = None

    # os.system('gdal_rasterize -ot Byte -a class -where "class=1" -tr ' + ' '.join([str(x) for x in [res, res]]) + ' -init 0 -a_nodata 0 ' + shpfile +' '+ shpfile.replace('.shp','.tif'))
    print(shpfile + " shp2geoTiff is finally!"+'\n')

def imgClipimg(imgfile,bigtif,outdir):
    '''
    employ gdal_translate is not resample,therefore crs is consistent
    '''
    outname = osp.basename(imgfile).split('_')[0]+'_img.tif'
    outfile = osp.join(outdir,outname)
    data = gdal.Open(imgfile, GA_ReadOnly)
    geoTransform = data.GetGeoTransform()
    minx = geoTransform[0]
    maxy = geoTransform[3]
    maxx = minx + geoTransform[1] * data.RasterXSize
    miny = maxy + geoTransform[5] * data.RasterYSize
    cmd1 = ['gdal_translate', "-ot", "Byte", "-of","GTiff", "-projwin",str(minx), str(maxy), str(maxx), str(miny),bigtif,outfile]
    print(' '.join(cmd1))
    subprocess.call(cmd1)
    print(outfile + " rasterClipraster is finally!"+'\n')

def reSamplebyimg(intimg,targetimg,outimg,pixsize=10):
    '''
    employ gdal_translate is not resample,therefore crs is consistent
    '''
    data = gdal.Open(targetimg, GA_ReadOnly)
    geoTransform = data.GetGeoTransform()
    minx = geoTransform[0]
    maxy = geoTransform[3]
    maxx = minx + geoTransform[1] * data.RasterXSize
    miny = maxy + geoTransform[5] * data.RasterYSize
    cmd1 = ['gdalwarp', "-tr", str(pixsize), str(pixsize), "-te" ,str(minx),str(miny), str(maxx), str(maxy), intimg, outimg]
    print(' '.join(cmd1))
    subprocess.call(cmd1)
    print(outimg + " reSamplebyimg is finally!"+'\n')

# img2tile 这一部分要进行拆分，image和lable的逻辑有区别，对于Nodata的指定（尤其是lable）
def img2tile(imgfile,out_file,offset_x,offset_y,clip_size=512,RGB=None):
    '''
    RGB is list, example [4,3,2]
    '''
    in_ds = gdal.Open(imgfile)
    bands_num = in_ds.RasterCount
    in_bands = []
    out_bands = []
    if RGB is None:
        for  band in range(1,bands_num+1):
            in_band = in_ds.GetRasterBand(band)
            in_bands.append(in_band)
            im_width = in_ds.RasterXSize-1
            im_height = in_ds.RasterYSize-1
            if im_width-offset_x <clip_size and im_height-offset_y < clip_size:
                out_band = np.zeros([clip_size,clip_size])
                out_band1 = in_band.ReadAsArray(offset_x, offset_y, im_width-offset_x, im_height-offset_y)
                out_band[:out_band1.shape[0],:out_band1.shape[1]] = out_band1[:,:]
            elif im_width-offset_x <clip_size:
                out_band = np.zeros([clip_size,clip_size])
                out_band1 = in_band.ReadAsArray(offset_x, offset_y, im_width-offset_x, clip_size)
                out_band[:out_band1.shape[0],:out_band1.shape[1]] = out_band1[:,:]
            elif im_height-offset_y < clip_size:
                out_band = np.zeros([clip_size,clip_size])
                out_band1 = in_band.ReadAsArray(offset_x, offset_y, clip_size, im_height-offset_y)
                out_band[:out_band1.shape[0],:out_band1.shape[1]] = out_band1[:,:]
            else:
                out_band = in_band.ReadAsArray(offset_x, offset_y, clip_size, clip_size)
            out_bands.append(out_band)
    else:
        for  band in RGB:
            in_band = in_ds.GetRasterBand(band)
            in_bands.append(in_band)
            im_width = in_ds.RasterXSize-1
            im_height = in_ds.RasterYSize-1
            if im_width-offset_x <clip_size and im_height-offset_y < clip_size:
                out_band = np.zeros([clip_size,clip_size])
                out_band1 = in_band.ReadAsArray(offset_x, offset_y, im_width-offset_x, im_height-offset_y)
                out_band[:out_band1.shape[0],:out_band1.shape[1]] = out_band1[:,:]
            elif im_width-offset_x <clip_size:
                out_band = np.zeros([clip_size,clip_size])
                out_band1 = in_band.ReadAsArray(offset_x, offset_y, im_width-offset_x, clip_size)
                out_band[:out_band1.shape[0],:out_band1.shape[1]] = out_band1[:,:]
            elif im_height-offset_y < clip_size:
                out_band = np.zeros([clip_size,clip_size])
                out_band1 = in_band.ReadAsArray(offset_x, offset_y, clip_size, im_height-offset_y)
                out_band[:out_band1.shape[0],:out_band1.shape[1]] = out_band1[:,:]
            else:
                out_band = in_band.ReadAsArray(offset_x, offset_y, clip_size, clip_size)
            out_bands.append(out_band)

    gtif_driver = gdal.GetDriverByName("GTiff")
    out_ds = gtif_driver.Create(out_file, clip_size, clip_size, bands_num, in_bands[0].DataType)
    ori_transform = in_ds.GetGeoTransform()

    top_left_x = ori_transform[0]  # x
    w_e_pixel_resolution = ori_transform[1] # pixel size of weight
    top_left_y = ori_transform[3] # y
    n_s_pixel_resolution = ori_transform[5] # pixel size of high

    top_left_x = top_left_x + offset_x * w_e_pixel_resolution
    top_left_y = top_left_y + offset_y * n_s_pixel_resolution

    dst_transform = (top_left_x, ori_transform[1], ori_transform[2], top_left_y, ori_transform[4], ori_transform[5])

    out_ds.SetGeoTransform(dst_transform)
    out_ds.SetProjection(in_ds.GetProjection())

    for num,out_band in enumerate(out_bands):
        num = num + 1
        out_ds.GetRasterBand(num).WriteArray(out_band)

    out_ds.FlushCache()
    del out_ds

if __name__ == "__main__":
    pass
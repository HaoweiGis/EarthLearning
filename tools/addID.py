from osgeo import ogr
import argparse
import os
import tqdm

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert shp to semantic segmentation datasets')
    parser.add_argument('--shp', default=r'‪D:\jinhua\Deeplearning_gendi\jindongSeg.shp' ,help='raster data path' ),

    parser.add_argument('--image', default=r'D:\EarthLearning\preprocessing\datasets\landcover\piont\SampleS2.tif' ,help='raster data path' )
    parser.add_argument('--output', default=r'D:\EarthLearning\preprocessing\datasets\landcover\piont', help='output path')
    args = parser.parse_args()
    return args


def featureAddId(shp_filename):
    ds=ogr.Open(shp_filename,1)
    lyr=ds.GetLayer()
    print('Sample feature number is: ',lyr.GetFeatureCount())
    fieldDefn = ogr.FieldDefn('objectId', ogr.OFTInteger)
    lyr.CreateField(fieldDefn)
    Ids = 0
    for feat in lyr:
        Ids = Ids + 1
        print(Ids)
        feat.SetField('objectId', Ids)
        lyr.SetFeature(feat)


def featureAddId(shp_filename):
    ds=ogr.Open(shp_filename,1)
    lyr=ds.GetLayer()
    print('Sample feature number is: ',lyr.GetFeatureCount())
    fieldDefn = ogr.FieldDefn('objectId', ogr.OFTInteger)
    lyr.CreateField(fieldDefn)
    Ids = 0
    for feat in lyr:
        Ids = Ids + 1
        print(Ids)
        feat.SetField('objectId', Ids)
        lyr.SetFeature(feat)

if __name__ == "__main__":
    args = parse_args()

    src_filename = args.image
    shp_filename = args.shp.replace('\u202a','')

    # featureAddId(shp_filename)

    


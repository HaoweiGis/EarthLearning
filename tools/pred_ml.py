import os,sys
import argparse
import numpy as np
import pickle
import pandas as pd

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

from geotools import geotools

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert shp to semantic segmentation datasets')
    parser.add_argument('--input', default=r'target.csv' ,help='raster data path' )
    parser.add_argument('--type', default='classification', 
    choices=['classification', 'regression', 'clustering',
            'denReduction', 'classificationCNN', 'regressionCNN'],
            help='classification,regression')

    parser.add_argument('--weight', default=r'work_dir\RFR\10-20_20-31-50.pkl', help='weight path')
    parser.add_argument('--output', default=r'result.csv', help='output path')
    args = parser.parse_args()
    return args

def pred_general_tif(input):
    im_shape = input.shape
    im_lines = None
    for i in range(im_shape[0]):
        im_line = input[i,:,:].flatten()
        if im_lines is None:
            im_lines = im_line
        else:
            im_lines = np.vstack((im_lines,im_line))
    im_lines = im_lines.transpose((1,0))

    model = pickle.load(open(args.weight, 'rb'))
    im_label = model.predict(im_lines).reshape(im_shape[1],im_shape[2])[np.newaxis,:, :]
    return im_label

def pred_general(input):
    df= pd.read_csv(input,header=None) 
    X = np.array(df.iloc[:,:])
    model = pickle.load(open(args.weight, 'rb'))
    im_label = model.predict(X)
    result = np.hstack((im_label[:,np.newaxis], X))
    return result

def pred_CNN(input):
    model = pickle.load(open(args.weight, 'rb'))
    im_shape = input.shape
    im_label = []
    for i in range(im_shape[1]):
        for j in range(im_shape[2]):
            pred_lines = input[:,i,j]
            lable = model.predict(pred_lines)
            im_label.append(lable)
    im_label = im_label.reshape(im_shape[1],im_shape[2])[np.newaxis,:, :]
    return im_label

def writefile(filename,lines):
    f = open(filename,'a')
    for line in lines:
        line = [str(i) for i in line]
        linestr = ','.join(line)
        f.writelines(linestr + '\n')

if __name__ == "__main__":
    args = parse_args()
    if os.path.splitext(args.input)[-1][1:] == "tif":

        im_data, im_porj, im_geotrans = geotools.GeoImgR(args.input)
        
        if args.type == 'classificationCNN':
            im_label = pred_CNN(im_data)
            geotools.GeoImgW(args.output, im_label, im_geotrans, im_porj)
        else:
            im_label = pred_general_tif(im_data)
            geotools.GeoImgW(args.output, im_label, im_geotrans, im_porj)
    else:
        if args.type == 'classificationCNN':
            csv_label = pred_CNN(args.input)
        else:
            csv_label = pred_general(args.input)
            writefile(args.output,csv_label)


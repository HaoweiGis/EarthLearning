import os,sys
import argparse
import datetime
import pickle

from numpy import core

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

from core.models.model_mls import * 
from geotools import geotools
from core.utils.logger import setup_logger

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert shp to semantic segmentation datasets')
    parser.add_argument('--input', default='sample.csv', help='support to byte')

    parser.add_argument('--type', default='regression', 
    choices=['classification', 'regression', 'clustering',
            'denReduction', 'classificationCNN', 'regressionCNN'],
            help='classification,regression')

    parser.add_argument('--model', default='RFR', help='support to byte')

    parser.add_argument('--output', default=r'work_dir', help='output path')
    args = parser.parse_args()
    return args


class TrainerML(object):
    def __init__(self, args):
        self.work_path = os.path.join(args.output,args.model)
        if not os.path.exists(self.work_path):
            os.makedirs(self.work_path)

        self.model_name = datetime.datetime.now().strftime("%m-%d_%H-%M-%S")
        self.logger = setup_logger("EarthLearning", self.work_path, filename='{}_{}_log.txt'.format(
            self.model_name, args.model ))
        self.logger.info(args)


    def ClassificationML(self):
        '''
        input(csv): row->n_samples; col->class,n_features
        output(pkl): path
        model: 'RF', 'SVM'
        '''
        # RF parameter default: n_estimators=1000 ...
        # SVM parameter default: ...
        model, reports = get_classificationML(input = args.input,  
        model=args.model)

        # model and log save
        model_path = os.path.join(self.work_path, self.model_name + '.pkl')
        pickle.dump(model, open(model_path, 'wb'))
        self.logger.info(reports)


    def RegressionML(self):
        '''
        input(csv): row->n_samples; col->regresionValue,n_features
        output(pkl): path
        model: 'RFR', 'SVR'
        '''
        # RFR parameter default: max_depth=2 ...
        # SVR parameter default: C=1.0, epsilon=0.2 ...
        model, reports= get_regressionML(input = args.input,  
        model=args.model)
        
        # model and log save
        model_path = os.path.join(self.work_path, self.model_name + '.pkl')
        pickle.dump(model, open(model_path, 'wb'))
        self.logger.info(reports)


    def ClusteringML(self):
        '''
        input(Geotif, csv): 
        output(pkl): path
        model: 'KMeans'
        '''
        # KMeans parameter default: n_clusters=10 ...
        im_data, _, _ = geotools.GeoImgR(args.input)
        model = get_clusteringML(input = im_data,  
        model=args.model)

        model_path = os.path.join(self.work_path, self.model_name + '.pkl')
        pickle.dump(model, open(model_path, 'wb'))
        self.logger.info("Clustering is OK! Path:" + model_path)


    def DReductionML(self):
        '''
        input(Geotif, csv):
        output(Geotif):
        model: 'PCA'
        '''
        # PCA parameter default: n_components = 1 ...
        im_data, im_porj, im_geotrans = geotools.GeoImgR(args.input)
        img_pac = get_dreductionML(input = im_data,  n_components=3,
        model=args.model)

        model_path = os.path.join(self.work_path, self.model_name + '.tif')
        geotools.GeoImgW(model_path, img_pac, im_geotrans, im_porj)
        self.logger.info("DReduction is OK! Path:" + model_path)


    def ClassificationCNN(self):
        pass

    def RegressionCNN(self):
        pass


if __name__ == "__main__":
    args = parse_args()

    if args.type == 'classification':
        Train = TrainerML(args)
        Train.ClassificationML()

    elif args.type == 'regression':
        Train = TrainerML(args)
        Train.RegressionML()

    elif args.type == 'clustering':
        Train = TrainerML(args)
        Train.ClusteringML()

    elif args.type == 'denReduction':
        Train = TrainerML(args)
        Train.DReductionML()

    elif args.type == 'classificationCNN':
        Train = TrainerML(args)
        Train.ClassificationCNN()

    elif args.type == 'regressionCNN':
        Train = TrainerML(args)
        Train.RegressionCNN()
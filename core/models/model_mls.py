from .machineLearning.classification import *
from .machineLearning.regression import *
from .machineLearning.clustering import *
from .machineLearning.dreduction import *

def get_classificationML(model, **kwargs):
    models = {
        'RF': get_randomforest,
        'SVM' : get_supportvector,
    }
    return models[model](**kwargs)

def get_regressionML(model, **kwargs):
    models = {
        'RFR': get_randomforestRegr,
        'SVR' : get_supportvectorRegr,
    }
    return models[model](**kwargs)

def get_clusteringML(model, **kwargs):
    models = {
        'KMeans': get_kmeans,
    }
    return models[model](**kwargs)
    

def get_dreductionML(model, **kwargs):
    models = {
        'PCA': get_principalcomponent,
    }
    return models[model](**kwargs)



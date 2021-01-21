import torch
import os
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from sklearn.decomposition import PCA
from sklearn import preprocessing



df= pd.read_csv(input,header=None) 
X = np.array(df.iloc[:,1:])
Y = np.array(df.iloc[:,[0]]).ravel()
x,y = sklearn.utils.shuffle(X, Y, random_state=1)
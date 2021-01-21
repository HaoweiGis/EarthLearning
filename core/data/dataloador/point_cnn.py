import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset
import sklearn
from sklearn import model_selection
import torch

def line2neighbor(line):
    neighbor = line.reshape((14,5,5))
    return neighbor

class CNNPointDataset(Dataset):

    BASE_NAME = 'input.csv'
    TestSize = 0.3

    def __init__(self, root=r'./datasets/', split='train', mode="Optical-SAR", transform=None, **kwargs):
        dataPath = os.path.join(root, self.BASE_NAME)
        df= pd.read_csv(dataPath) 
        X = np.array(df.iloc[:,1:])
        Y = np.array(df.iloc[:,[0]]).ravel()
        x,y = sklearn.utils.shuffle(X, Y, random_state=1)
        x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=self.TestSize, random_state=42)

        if split == 'train':
            self.images = x_train
            self.masks = y_train
        elif split == 'test':
            self.images = x_test
            self.masks = y_test

    def __getitem__(self, index):
        image = self.images[index]
        label = np.array(self.masks[index]).astype('int32')
        neighbor = line2neighbor(image)
        neighbor = torch.from_numpy(neighbor).long()
        label = torch.from_numpy(label).long()
        return (neighbor, label)
        
    def __len__(self):
        return len(self.images)
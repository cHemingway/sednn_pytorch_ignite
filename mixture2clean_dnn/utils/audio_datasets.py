''' Audio Dataset - Pytorch compatible datasets for the speech files '''

import pickle

import torch
from torch.utils.data import Dataset

from utils.utilities import (load_hdf5)

class NoisySpeechFeaturesDataset(Dataset):
    ''' Dataset for noisy speech features only '''

    def __init__(self, features_path, scaler_path):
        '''
        Args:
            features_path: (string/file): h5 file with features
            scaler_path: .p file with scalar
        '''
        # Load x and y as tensors
        (self.x, self.y) = load_hdf5(features_path)
        (_, self.n_concat, self.freq_bins) = self.x.shape
        self.x = torch.from_numpy(self.x)
        self.y = torch.from_numpy(self.y)

        # Load scaler params
        scaler = pickle.load(open(scaler_path, 'rb'))
        self.scale_mean = torch.from_numpy(scaler['mean'])
        self.scale_std = torch.from_numpy(scaler['std'])

        # Scale, ignoring autograd
        # TODO: use torchvision Normalize instead?
        with torch.no_grad():
            self.scale(self.x)
            self.scale(self.y)

    def scale(self, x):
        ''' Scale value in place by scalar '''
        x.sub_(self.scale_mean) # Todo operate on last dimension
        x.div_(self.scale_std)

    def __len__(self):
        return len(self.x) # 1st dimension

    def __getitem__(self, idx):
        """Get a specific input x, output y
        
        Arguments:
            idx {int} -- Index of pair
        
        Returns:
            x,y -- Pytorch 
        """
        # TODO Check indexes
        return self.x[idx,], self.y[idx,]


    def get_properties():
        ''' Returns a directory of the dataset properties  
        Usage:
            DNN(... . get_properties())
        '''
        return {"n_concat":self.n_concat, "freq_bins":self.freq_bins}

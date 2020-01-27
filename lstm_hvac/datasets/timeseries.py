from torch.utils import data
from scipy import stats

import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
from itertools import repeat, chain
import torchvision.transforms as T
import lstm_hvac.preprocessing as proc

class TimeSeriesDataset(data.Dataset):
    def __init__(self, data, target_idx,
                mode='train',
                split_v=[.7, .3, 0.0],
                timesteps=50, 
                horizon=50, 
                output_size=1,
                input_buffer=1):
        super(TimeSeriesDataset, self).__init__()   
        self.data = data 
        self.mode = mode
        self.target_idx = target_idx
        self.split_v = split_v
        self.timesteps = timesteps
        self.horizon = horizon
        self.output_size = output_size
        self.input_buffer = input_buffer
        self.split_ids = tuple(map(lambda x: int(x), {
            'train' : (0, self.split_v[0] * self.data.shape[0]),
            'test' : (self.split_v[0] * self.data.shape[0], -1),
        }[self.mode]))    
        self.data = data[self.split_ids[0]:self.split_ids[1]]

        self.transform = T.Compose([
            proc.WienerFilter(n=200),
            proc.Downsample(15),
            proc.Normalize(apply_log=False), 
        ])
        self.data = self.transform(self.data)

    def __getitem__(self, idx):
        x = np.empty((0, self.data.shape[-1], self.input_buffer))
        y = np.empty((0, self.output_size)) 
        for t in range(self.timesteps):
            x = np.vstack((x, self.data[idx + t : idx + t + self.input_buffer].transpose()[np.newaxis]))
            y = np.vstack((y, self.data[np.newaxis, idx + t + self.input_buffer + self.horizon:\
                                    idx + t + self.input_buffer + self.horizon + self.output_size, self.target_idx]))
        return (x.astype(float), y.astype(float))

    def __len__(self):
        return  (self.data.shape[0] - self.timesteps - self.horizon - self.input_buffer - self.output_size)

    @property
    def n_features(self):
        return self.data.shape[1] * self.input_buffer

    
class DatasetHVAC(TimeSeriesDataset):
    def __init__(self, *args,**kwargs):
        self.dir = 'lstm_hvac/data/hvac/dataHVAC.npy'
        data = np.load(self.dir, allow_pickle=True)[:, 2:].astype('float')
        super(DatasetHVAC, self).__init__(data, 4, *args, **kwargs)
        del data


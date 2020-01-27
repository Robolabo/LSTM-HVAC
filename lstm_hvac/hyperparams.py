
import os
import argparse
import json
import torch

class Hyperparams(object):
    
    def __init__(self, json,
                 cuda=True, 
                 chk_file='chk',
                 predictor='MSPM2',
                 epochs=20, 
                 batch_size=64, 
                 timesteps=200, 
                 horizon=200,  
                 ann_units=[100, 100, 100], 
                 optimizer='adam', 
                 learning_rate=1e-3, 
                 momentum=0.0):
            self.json = json
            self.chk_file = chk_file
            self.predictor = predictor
            self.device = torch.device('cuda:0' if cuda else 'cpu')
            self.epochs = epochs
            self.init_epoch = 0
            self.batch_size = batch_size
            self.timesteps = timesteps
            self.horizon = horizon
            self.ann_units = ann_units
            self.optimizer = optimizer
            self.learning_rate = learning_rate
            self.momentum = momentum
            self.input_buffer = 1
            self.output_size = 1
            self.loss = None
            self.spl = None
            
            if json is not None:
                self.parse_json()

    def parse_json(self):
        json_path = os.path.join('lstm_hvac', 'Settings', self.json + '.json')
        if os.path.exists(json_path):
            with open(json_path) as json_file:
                self.__dict__.update(json.load(json_file))

    def __repr__(self):
        repr_str = type(self).__name__ + '(\n'
        return repr_str + ''.join(['\t' + str(k) + ' : ' + str(v) + '\n'\
             for k, v in self.__dict__.items() if k is not 'net_dict']) + ')'

    def as_dict(self):
        return self.__dict__
        
    def update(self, **kwargs):
        self.__dict__.update(**kwargs)
    




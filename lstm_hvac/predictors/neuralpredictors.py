import pdb
import math
import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.utils import weight_norm
from functools import reduce
from collections import deque, OrderedDict
from itertools import chain, repeat
from lstm_hvac.predictors import BasePredictor
torch.backends.cudnn.benchmark = True


class MSPM1(BasePredictor):
    def __init__(self, *args, **kwargs):
        super(MSPM1, self).__init__(*args, **kwargs)
        # recurrent layers
        self.lstm1 = nn.LSTM(self.input_size, self.units[0], num_layers=2, batch_first=True)
        # add non-rec layers
        self.out_mlp = nn.Sequential(
            nn.Linear(self.units[0], self.output_size),
        )
        self.init_optimizer()
        self.to(self.device)

    def forward(self, seq, states):
        out1, states_out = self.lstm1(seq, states)
        y_tot = self.out_mlp(out1[:,-1])
        return y_tot, states_out
    
    def init_states(self):
        return (torch.zeros([2, 1, self.units[0]], dtype=torch.float).to(self.device),
                torch.zeros([2, 1, self.units[0]], dtype=torch.float).to(self.device))

class MSPM2(BasePredictor):
    def __init__(self, *args, **kwargs):
        super(MSPM2, self).__init__(*args, **kwargs)
        # recurrent layers
        self.lstm1 = nn.LSTM(self.input_size, self.units[0], num_layers=2, batch_first=True)
        # add non-rec layers
        self.out_mlp = nn.Sequential(
            nn.Linear(self.units[0], 1),
        )
        self.init_optimizer()
        self.to(self.device)

    def forward(self, seq, states):
        out1, states_out = self.lstm1(seq, states)
        y_tot = self.out_mlp(out1)
        return y_tot, states_out
    
    def init_states(self):
        return (torch.zeros([2, 1, self.units[0]], dtype=torch.float).to(self.device),
                torch.zeros([2, 1, self.units[0]], dtype=torch.float).to(self.device))

class EncDec(BasePredictor):
    def __init__(self, *args, **kwargs):
        super(EncDec, self).__init__(*args, **kwargs)
        # recurrent layers
        self.encoder = nn.LSTM(self.input_size, self.units[0], num_layers=1, batch_first=True)  
        self.decoder = nn.LSTM(1, self.units[1], num_layers=1, batch_first=True)

        self.linear_c = nn.Linear(self.units[0], self.units[1])
        self.linear_h = nn.Linear(self.units[0], self.units[1])

        self.head = nn.Linear(self.units[1], 1)
        self.init_optimizer()
        self.to(self.device)

    def forward(self, seq, states, ignore_decoder=False):
        enc_out, ctx = self.encoder(seq, states)
        decoder_state = (self.linear_c(ctx[0][-1]).unsqueeze(0), self.linear_h(ctx[1][-1]).unsqueeze(0))
        decoder_input = torch.tensor(0.0).to(self.device).view(1,1,1)
        y_tot = []
        if ignore_decoder:
            return y_tot, ctx
        for _ in range(self.output_size):  
            _, decoder_state = self.decoder(decoder_input, decoder_state)
            y = self.head(decoder_state[0])
            y_tot.append(y)
            decoder_input = y
        return torch.cat(y_tot), ctx

    def init_states(self):
        return (torch.zeros([1, 1, self.units[0]], dtype=torch.float).to(self.device),
                torch.zeros([1, 1, self.units[0]], dtype=torch.float).to(self.device))

        
        


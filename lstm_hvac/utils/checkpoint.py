
import os
import json
import numpy as np
from argparse import Namespace
import matplotlib
import matplotlib.pyplot as plot
import seaborn as sns
sns.set(color_codes=True)
import pdb
import torch
import json
from ast import literal_eval
from lstm_hvac import predictors as pred

def save_checkpoint(model, epoch, hyperparams, loss_hist, ModelConnection=None):
	hyperparams.update(**{'init_epoch' : epoch})
	checkpoint={'state_dict' : model.state_dict(),
				'optimizer' : model.optimizer.state_dict(),
				'hyperparams' : hyperparams.as_dict(),
				'model_name' : model.__class__.__name__,
				'net_input_size' : model.input_size,
				'loss_hist' : loss_hist
				}

	path = os.path.join('lstm_hvac', 'logs', 'models', hyperparams.chk_file + '.pth.tar') 
	torch.save(checkpoint, path)

def load_checkpoint(model, hyperparams):
	path = os.path.join('lstm_hvac', 'logs', 'models', hyperparams.chk_file + '.pth.tar') 
	print(path)
	# enables to run gpu model on cpu
	checkpoint = torch.load(path, map_location={'cuda:0': 'cpu'})
	new_epochs = hyperparams.epochs
	hyperparams.update(**checkpoint['hyperparams'])
	hyperparams.update(epochs = new_epochs)
	
	# restore model parameters and optimizer
	model.load_state_dict(checkpoint['state_dict'])
	model.optimizer.load_state_dict(checkpoint['optimizer'])

	# set model config
	model.__dict__.update({
		'batch_size' : hyperparams.batch_size,
		'timesteps' : hyperparams.timesteps,
		'horizon' : hyperparams.horizon,
		'input_size' : checkpoint['net_input_size'],
	})
	# model.setUnits(hyperparams.ann_units)
	loss_hist = checkpoint['loss_hist']
	return model, hyperparams, loss_hist

	


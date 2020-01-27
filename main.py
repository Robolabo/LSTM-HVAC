# general purpose imports
import pdb
import click
import numpy as np
from torch.utils import data

# ts_pred imports
import lstm_hvac
import lstm_hvac.predictors as pred
from lstm_hvac.hyperparams import Hyperparams
from lstm_hvac.datasets import DatasetHVAC

@click.command()
@click.option('--cuda', default=True, is_flag=True, help='')
@click.option('--resume', default=False, is_flag=True,  help='')
@click.option('--predict', default=False, is_flag=True,  help='')
@click.option('--json', default=None, help='Name of the json file storing hyperparams or None if via click args.')    
@click.option('--chk_file', default='chk', help='Name file to save training checkpoint.')    
def main(resume, predict, **kwargs):
    # Hyperparameter object
    hyps = Hyperparams(**kwargs)

    #* Data Loaders
    # TRAIN DATA LOADER
    train_generator = data.DataLoader(DatasetHVAC(mode='train', **{
        'timesteps' : hyps.timesteps,
        'output_size' : hyps.output_size,
        'input_buffer' : 1,
        'horizon' : hyps.horizon,
        'split_v' : hyps.spl
    }), batch_size=1, drop_last=True)

    # TEST DATA LOADER
    test_generator = data.DataLoader(DatasetHVAC(mode='test', **{
        'timesteps' : 1, 
        'output_size' : hyps.output_size,
        'input_buffer' : 1,
        'horizon' : hyps.horizon,
        'split_v' : hyps.spl
    }), batch_size=1)

    model_class = {
        "MSPM1" : pred.MSPM1,
        "MSPM2" : pred.MSPM2,
        "EncDec" : pred.EncDec 
    }[hyps.predictor]
    
    # NEURAL NETWORK
    rnn = model_class(**{
        'units' : hyps.ann_units,
        'learning_rate' : hyps.learning_rate,
        'batch_size' : hyps.batch_size,
        'timesteps' : hyps.timesteps,
        'horizon' : hyps.horizon,
        'input_size' :  train_generator.dataset.n_features, 
        'device' : hyps.device,
        'output_size': hyps.output_size
    })
    # dict to store the loss history of training
    loss_hist = {'train': np.array([]), 'val': np.array([])}

    # restore previous training
    if resume or predict:
        rnn, hyps, loss_hist = lstm_hvac.utils.load_checkpoint(rnn, hyps)
    
    if not predict:
        #* Train the Model
        rnn, loss_hist = rnn.train(hyps.epochs, train_generator, hyps, init_epoch=hyps.init_epoch)
    
    # evaluate the model
    rnn.predict(test_generator)

if __name__ == '__main__':
    main()

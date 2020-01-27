import pdb
import time
import torch 
import numpy as np
import torch.nn as nn
import torch.optim as optim

from itertools import repeat
import matplotlib.pyplot as plot 
from lstm_hvac.utils import save_checkpoint
from lstm_hvac.metrics import mse, rmse

class BasePredictor(nn.Module):
    def __init__(self, units, is_recurrent=True, optimizer='adam', learning_rate=1e-3, \
                    batch_size=1, timesteps=50, input_size=1, horizon=50, 
                    output_size=1, device="cuda:0"):
        super(BasePredictor, self).__init__()  
        self.units = units
        self.is_recurrent = is_recurrent
        self.optimizer = None   
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.timesteps = timesteps
        self.input_size = input_size
        self.horizon = horizon
        self.device = device
        self.output_size = output_size
        self.loss_fn = mse 
        
    def forward(self, x):
        raise NotImplementedError
    
    def train(self, epochs, generator, hyperparams, init_epoch=0, loss_hist=None, gen_val=None):   
        if self.optimizer is None:
            self.init_optimizer()
        loss_hist = ({'train': np.array([]), 'val': np.array([])}, loss_hist)[loss_hist is not None]
        for epoch in range(init_epoch, epochs):
            epoch_loss = 0
            t0 = time.time()
            states =  self.init_states()
            loss = torch.tensor(0.0).to(self.device)
            for step, (inputs, targets) in enumerate(generator): 
                inputs, targets = inputs.float().to(self.device), targets.to(self.device).float()
                inputs = inputs.squeeze().unsqueeze(0)
                if type(self).__name__ == 'EncDec':
                    _, states = self(inputs[0,0].view(1,1,-1),  states, ignore_decoder=True)
                else:
                    _, states = self(inputs[0,0].view(1,1,-1), states)
                predictions, _ = self(inputs[0,1:].unsqueeze(0), states)
                if type(self).__name__ == 'EncDec':
                    loss += self.loss_fn(predictions.view(-1), targets[:,-1].view(-1)) 
                elif type(self).__name__ == 'MSPM1':
                    loss += self.loss_fn(predictions.view(-1), targets[:, -1].view(-1)) 
                else:
                    loss += self.loss_fn(predictions[0,-1].view(-1), targets[:, -1].view(-1)) 

                if step % self.batch_size == 0:
                    loss /= self.batch_size
                    self.optimizer.zero_grad()
                    # loss = self.apply_regularization(loss)
                    loss.backward(retain_graph=True)
                    self.optimizer.step()
                    
                    print("epoch -> {}/{}, step -> {}/{}, loss -> {:.3E}".format(int(epoch), \
                            epochs, int(step), len(generator), loss.item()), end='\r')
                    epoch_loss += loss.item()
                    loss = torch.tensor(0.0).to(self.device)
                states = [st.detach_() for st in states]
                  
            epoch_loss /= (len(generator) // self.batch_size)
            loss_hist['train'] = np.append(loss_hist['train'], epoch_loss)
            save_checkpoint(self, epoch + 1, hyperparams, loss_hist, None)
            print("epoch -> {}/{}, loss -> {:.3E}, time_elapsed -> {:.3E}".format(int(epoch),\
                epochs, epoch_loss, time.time()-t0))
        return self, loss_hist

    def predict(self, generator):
        with torch.no_grad():
            eval_loss = 0
            previous_prediction = None
            # assign same to all vars at once
            predictions = torch.tensor([]).float().to(self.device)
            targets = torch.tensor([]).float().to(self.device)
            inputs = torch.tensor([]).float().to(self.device)

            states = self.init_states() 
            for step, (x, y) in enumerate(generator):
                x, y = x.float().to(self.device), y.float().to(self.device)
                x = x.view([1,1,-1])
                output, states = self(x, states)   
                output = output.squeeze()
                y = y.squeeze()
                predictions = torch.cat((predictions, output.unsqueeze(0)), 0)
                targets = torch.cat((targets, y.unsqueeze(0)), 0)
                #* Stack is stored in GPU to make a faster loop (may have VRAM issues!)
                print("Prediction step {}/{}".format(step, len(generator)), end='\r')
            
            targets = targets.cpu().squeeze().numpy()
            predictions = predictions.cpu().squeeze().numpy()
      
            if type(self).__name__ == 'MSPM2':
                predictions = predictions[:,np.newaxis]
                targets = targets[:,np.newaxis]

            test_dict = {
                'MSE' : np.mean((predictions[:,-1] - targets[:,-1])**2),
                'NRMSE' : np.sqrt(np.mean((predictions[:,-1] - targets[:,-1])**2)),
                'MAE' : np.mean(np.absolute(predictions[:,-1] - targets[:,-1]))
            }
            plot.plot(predictions[:,-1])
            plot.plot(targets[:,-1])
            plot.legend(['Predictions', 'Targets'])
            plot.show()
        
        return predictions, test_dict

    def apply_regularization(self, loss, lambda_factor=.5):
        regularization_term = torch.tensor(0.).cuda()
        for param in self.parameters():
            regularization_term += torch.norm(param)
        return loss + lambda_factor * regularization_term

    
    def init_optimizer(self):
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
    

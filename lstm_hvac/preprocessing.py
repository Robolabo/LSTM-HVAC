import pdb
import torch
import numpy as np
import torchvision.transforms as T
import scipy.signal as signal 


class Normalize(object):
    def __init__(self, min_v=None, max_v=None, apply_log=False):
        self.fn = lambda x: (isinstance(x, torch.Tensor) and x.log() or np.log(x)) \
                                if apply_log else x
        self.min_v = min_v and self.fn(min_v) or min_v
        self.max_v = max_v and self.fn(max_v) or max_v

    def __call__(self, x):
        min_vals = (self.min_v, self.fn(x.min(0)))[self.min_v is None]
        max_vals = (self.max_v, self.fn(x.max(0)))[self.max_v is None] 
        
        return (self.fn(x) - min_vals) / (max_vals - min_vals + 1e-5)

class QuantileNorm(object):
    def __call__(self, x):
      quantiles = np.quantile(x, [.25, .75], axis=0)
      iqr = quantiles[1] - quantiles[0]
      max_v = quantiles[1] + 1.5 * iqr
      min_v = quantiles[0] - 1.5 * iqr
      return (x - min_v) / (max_v - min_v)


class Standarize(object):
    def __init__(self, mean_v=None, std_v=None, eps=1e-7):
        self.mean_v = mean_v
        self.std_v = std_v
        self.eps = eps
      
    def __call__(self, x):
        mean_vals = (self.mean_v, x.mean(0))[self.mean_v is None]
        std_vals = (self.std_v, x.std(0))[self.std_v is None] 
        return (x - mean_vals + .5) / (std_vals + self.eps)

class Downsample(object):
    def __init__(self, sampling_period):
        self.sampling_period = sampling_period

    def __call__(self, x):
      return x[::self.sampling_period]
      y = np.stack([x[i:i + self.sampling_period].mean(0) \
            for i in np.arange(0, x.shape[0], self.sampling_period)])
      return y
      
class Differenciate(object):
    def __call__(self, x):
      return x[1:] - x[:-1]
    

class WienerFilter(object):
    def __init__(self, n, noise_power=None):
        self.n=n
        self.noise_power = noise_power
    def __call__(self, x):
        y = x.copy()
        for i in range(x.shape[1]):
            y[:, i] = signal.wiener(x[:, i], self.n, self.noise_power)
        return y
  
class MovingAverage(object):
  def __init__(self, n):
    self.n = n
  def __call__(self, x):
    return np.stack([np.convolve(ts, np.ones([self.n]) / self.n) \
                    for ts in x.transpose(1, 0)], 1)[self.n:-self.n]
  
  
class RemoveOutliers(object):
    """ Does not support adaptive preprocessing """
    #TODO : deal with cat data and impl use_diff
    def __init__(self, q1, q2, num_ids=None, use_diff=False):
      self.q1 = q1
      self.q2 = q2
      self.num_ids = num_ids
      self.use_diff = False  #! not implem yet

    def __call__(self, x):
      quantiles = np.quantile(x[:, self.num_ids], [self.q1, self.q2], axis=0)
      iqr = quantiles[1] - quantiles[0]
      
      return np.stack(tuple(filter(lambda v: all(v[self.num_ids] > quantiles[0] - 1.5 * iqr)\
                            and all(v[self.num_ids] < quantiles[1] + 1.5 * iqr), x))) 


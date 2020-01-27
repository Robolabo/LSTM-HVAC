
import numpy as np
import torch

def mse(pred, target):
    "Mean square error metric"
    if type(pred).__name__ is 'Tensor':
        return (pred - target).pow(2).mean()
    else:
        return np.power(pred - target, 2).mean()


def mae(pred, target):
    "Mean absolute error metric"
    if type(pred).__name__ is 'Tensor':
        return (pred - target).abs().mean()
    else:
        return np.absolute(pred - target).mean()


def rmse(pred, target):
    "Root mean square error metric"
    if type(pred).__name__ is 'Tensor':
        return mse(pred, target).sqrt()
    else:
        return np.sqrt(mse(pred, target))

def mape(pred, target, eps=1e-5):
    "Mean absolute percentage error"
    if type(pred).__name__ is 'Tensor':
        return ((target - pred) / (target + eps)).abs().mean() * 100
    else:
        return np.absolute(np.divide(target - pred, target + eps)).mean()

def mase(pred, target, m=49):
    """
    Mean absolute scaled error 
    Measures how good is our prediction compared to a naive prediction (y(t) = y(t-1))
    paper: https://www.sciencedirect.com/science/article/pii/S0169207006000239
    """
    return mae(pred, target) / mae(target[..., 1:], target[..., 0].unsqueeze(-1))

def geo_mean(x):
    """ Computes the geometric mean of a torch tensor or numpy array"""
    if type(x).__name__ is 'Tensor':
        return x.prod().pow(1 / x.shape[0])
    else:
        return np.power(x.prod(), x.shape[0])

def gmse(pred, target):
    "Geometric mean squere error"
    if type(pred).__name__ is 'Tensor':
        return geo_mean((pred - target).pow(2))
    else:
        return geo_mean(np.power(pred - target, 2))

def sae(pred, target):
    """ Sum of absolute errors """
    if type(pred).__name__ is 'Tensor':
        return (pred - target).abs().sum()
    else:
        return np.absolute(pred - target).sum()

def pearson_coef(pred, target):
    pass
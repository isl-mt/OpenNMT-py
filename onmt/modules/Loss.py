import torch
import torch.nn as nn

# Implementation of loss functions if needed

# Mean Square Error
def mse_loss(input, target):
    return torch.sum((input - target) ** 2)

# Weighted min square error
def weighted_mse_loss(input, target, weight):
    return torch.sum(weight * (input - target) ** 2)

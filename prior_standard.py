import torch

import torch.nn as nn
import torch.nn.functional as F

import numpy as np

PI = torch.from_numpy(np.asarray(np.pi))

def log_standard_normal(x, reduction=None, dim=None):
    D = x.shape[1]
    log_p = -0.5 * D * torch.log(2. * PI) - 0.5 * x**2.
    if reduction == 'avg':
        return torch.mean(log_p, dim)
    elif reduction == 'sum':
        return torch.sum(log_p, dim)
    else:
        return log_p

class Prior(nn.Module):
    def __init__(self, length=2):
        super(Prior, self).__init__()

        self.length = length 

        # params weights
        self.means = torch.zeros(1, length)
        self.logvars = torch.zeros(1, length)

    def get_params(self):
        return self.means, self.logvars

    def sample(self, batch_size):
        return torch.randn(batch_size, self.length)
    
    def log_prob(self, z):
        return log_standard_normal(z)
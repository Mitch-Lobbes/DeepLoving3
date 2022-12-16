import torch

import torch.nn as nn

import numpy as np

PI = torch.from_numpy(np.asarray(np.pi))
EPS = 1.e-5

def log_standard_normal(x, reduction=None, dim=None):
    D = x.shape[1]
    log_p = -0.5 * D * torch.log(2. * PI) - 0.5 * x**2.
    if reduction == 'avg':
        return torch.mean(log_p, dim)
    elif reduction == 'sum':
        return torch.sum(log_p, dim)
    else:
        return log_p

def log_normal_diag(x, mu, log_var, reduction=None, dim=None):
    D = x.shape[1]
    log_p = -0.5 * D * torch.log(2. * PI) - 0.5 * log_var - 0.5 * torch.exp(-log_var) * (x - mu)**2.
    if reduction == 'avg':
        return torch.mean(log_p, dim)
    elif reduction == 'sum':
        return torch.sum(log_p, dim)
    else:
        return log_p

# YOUR CODE GOES HERE
# This class combines Encoder, Decoder and Prior.
# NOTES:
# (i) The function "sample" must be implemented.
# (ii) The function "forward" must return the negative ELBO. Please remember to add an argument "reduction" that is either "mean" or "sum".
class VAE(nn.Module):
    def __init__(self, encoder, decoder, prior, latent_dim):
        super(VAE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.prior = prior

        self.latent_dim = latent_dim

    def forward(self, x, reduction='sum'):
        mu, log_var = self.encoder.encode(x)
        z = self.encoder.reparameterization(mu, log_var)

        # Negative ELBO (NELBO)
        RE = self.decoder.forward(z, x, reduction=reduction)
        KL = log_normal_diag(z, mu, log_var, reduction=reduction) - log_standard_normal(z, reduction=reduction)
        NELBO = -(RE - KL)

        if reduction == 'sum':
            return NELBO.sum()
        else:
            return NELBO.mean()

    def sample(self, batch_size=64):
        z = self.prior.sample(batch_size)
        return self.decoder.sample(z)
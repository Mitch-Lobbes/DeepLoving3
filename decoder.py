import torch

import torch.nn as nn

import numpy as np

# YOUR CODE GOES HERE
# NOTE: The class must containt the following function: 
# (i) sample
# Moreover, forward must return the log-probability of the conditional likelihood function for given z, i.e., log p(x|z)

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

class Decoder(nn.Module):
    def __init__(self, img_channels, img_size, latent_dim):
        super(Decoder, self).__init__()
        num_features = 32 * img_size * img_size
        self.img_channels = img_channels
        self.img_size = img_size
        self.fc1 = nn.Linear(latent_dim, num_features)
        self.conv1 = nn.ConvTranspose2d(32, 16, 5, padding=2)
        self.conv2 = nn.ConvTranspose2d(16, img_channels, 5, padding=2)

    def decode(self, z):
        model = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            nn.Unflatten(1, (32, self.img_size, self.img_size)),
            self.conv1,
            nn.ReLU(),
            self.conv2,
            nn.Sigmoid()
        )
        
        x_hat = model(z)

        return x_hat

    def sample(self, z):
        """
        For a given latent code compute parameters of the conditional likelihood 
        and sample x ~ p(x|z)

        z: torch.tensor, with dimensionality (mini-batch, z_dim)

        return:
        x: torch.tensor, with dimensionality (mini-batch, x_dim)
        """
        x = torch.zeros((z.shape[0], self.img_channels, self.img_size, self.img_size))
        
        return self.forward(z, x)

    def forward(self, z, x, reduction='sum'):
        """
        Compute the log probability: log p(x|z). 
        z: torch.tensor, with dimensionality (mini-batch, z_dim)
        x: torch.tensor, with dimensionality (mini-batch, x_dim)
        """
        x_hat = self.decode(z)

        log_p = log_normal_diag(x, x_hat, log_var=torch.Tensor([0.0]), reduction=reduction)

        return log_p
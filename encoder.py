import torch

import torch.nn as nn

# YOUR CODE GOES HERE
# NOTE: The class must containt the following functions: 
# (i) reparameterization
# (ii) sample
# Moreover, forward must return the log-probability of variational posterior for given x, i.e., log q(z|x)

class Encoder(nn.Module):
    def __init__(self, img_channels, img_size, z_dim):
        super(Encoder, self).__init__()
        num_features = 32 * img_size * img_size
        self.conv1 = nn.Conv2d(img_channels, 16, 5, padding=2)
        self.conv2 = nn.Conv2d(16, 32, 5, padding=2)
        self.fc_mu = nn.Linear(num_features, z_dim)
        self.fc_var = nn.Linear(num_features, z_dim)

    @staticmethod
    def reparameterization(mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std) # get values from an independent normal distribution ~N(0,1)
        return mu + std * eps

    def sample(self, x=None, mu_e=None, log_var_e=None):
        """
        Sample from the encoder. 
        If x is given (not equal to None), then copmute variational posterior distribution q(z|x) and sample from it.
        Otherwise, use `mu_e` and `log_var_e` as parameter of the variational posterior and sample from it.

        x: torch.tensor, with dimensionality (mini-batch, x_dim)
             a mini-batch of data points
        mu_e: torch.tensor, with dimensionality (mini-batch, x_dim)
             mean vector of the variational posterior
        log_var_e: torch.tensor, with dimensionality (mini-batch, x_dim)
             log variance of the variational posterior
        return: z: torch.tensor, with dimensionality (mini-batch, z_dim)
        """
        if x is not None:
            return self.forward(x)
        else:
            return self.reparameterization(mu_e, log_var_e)

    def get_log_probs(self, x):
        model = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            self.conv2,
            nn.ReLU(),
            nn.Flatten()
        )
        conv_output = model(x)
        mu = self.fc_mu(conv_output)
        log_var = self.fc_var(conv_output)

        return mu, log_var

    def forward(self, x):
        """
        Compute log-probability of variational posterior for given x, i.e., log q(z|x)
        x: torch.tensor, with dimensionality (mini-batch, x_dim)
             a mini-batch of data points
        """
        mu, log_var = self.get_log_probs(x)
        
        return self.reparameterization(mu, log_var)
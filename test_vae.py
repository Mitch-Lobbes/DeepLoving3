from encoder import Encoder
from decoder import Decoder
from prior_standard import Prior
from vae import VAE
import torch

class TestVAE:
    def __init__(self) -> None:
        self.img_channels = 1
        self.img_size = 28
        self.latent_dim = 256
        self.batch_size = 6
        self.encoder = Encoder(self.img_channels, self.img_size, self.latent_dim)
        self.decoder = Decoder(self.img_channels, self.img_size, self.latent_dim)
        self.prior = Prior(length=self.latent_dim)
        self.vae = VAE(self.encoder, self.decoder, self.prior, self.latent_dim)

    def test_output_shapes(self):
        x = torch.zeros((self.batch_size, self.img_channels, self.img_size, self.img_size))
        nelbo = self.vae.forward(x)
        sample = self.vae.sample(batch_size=self.batch_size)

        assert nelbo.shape == ()
        assert sample.shape == (self.batch_size, self.img_channels, self.img_size, self.img_size)

if __name__=='__main__':
    test_vae = TestVAE()
    test_vae.test_output_shapes()


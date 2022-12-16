from encoder import Encoder
import torch

class TestEncoder:
    def __init__(self) -> None:
        self.img_channels = 1
        self.img_size = 28
        self.latent_dim = 256
        self.encoder = Encoder(self.img_channels, self.img_size, self.latent_dim)
        self.batch_size = 6

    def test_output_shapes(self):
        x = torch.zeros((self.batch_size, self.img_channels, self.img_size, self.img_size))
        mu, log_var = self.encoder.encode(x)
        z1 = self.encoder.reparameterization(mu, log_var)
        z2 = self.encoder.forward(x)

        assert mu.shape == (self.batch_size, self.latent_dim)
        assert log_var.shape == (self.batch_size, self.latent_dim)
        assert z1.shape == (self.batch_size, self.latent_dim)
        assert z2.shape == (self.batch_size, self.latent_dim)

if __name__=='__main__':
    test_encoder = TestEncoder()
    test_encoder.test_output_shapes()


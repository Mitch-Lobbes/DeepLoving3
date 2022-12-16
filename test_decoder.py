from decoder import Decoder
import torch

class TestDecoder:
    def __init__(self) -> None:
        self.img_channels = 1
        self.img_size = 28
        self.latent_dim = 256
        self.decoder = Decoder(self.img_channels, self.img_size, self.latent_dim)
        self.batch_size = 6

    def test_output_shapes(self):
        x = torch.zeros((self.batch_size, self.img_channels, self.img_size, self.img_size))
        z = torch.zeros((self.batch_size, self.latent_dim))
        x_hat = self.decoder.decode(z)
        log_p = self.decoder.forward(z, x)

        assert x_hat.shape == (self.batch_size, self.img_channels, self.img_size, self.img_size)
        assert log_p.shape == (self.batch_size, self.img_channels, self.img_size, self.img_size)
        
if __name__=='__main__':
    test_decoder = TestDecoder()
    test_decoder.test_output_shapes()


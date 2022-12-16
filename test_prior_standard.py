from prior_standard import Prior
import torch

class TestPrior:
    def __init__(self) -> None:
        self.length = 2
        self.batch_size = 6
        self.latent_dim = 265
        self.prior = Prior(self.length)

    def test_output_shapes(self):
        z = torch.zeros((self.batch_size, self.latent_dim))
        log_prob = self.prior.log_prob(z)
        sample = self.prior.sample(self.batch_size)

        assert log_prob.shape == z.shape
        assert sample.shape == (self.batch_size, self.length)
        
if __name__=='__main__':
    test_prior = TestPrior()
    test_prior.test_output_shapes()


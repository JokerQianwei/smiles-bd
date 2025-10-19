
import torch

class ClippedLinearSchedule:
    """
    Sample mask rates in [beta, omega].
    Weight ~ (omega - beta)/(mask_rate + eps)  (∝ α'(t)/(1-α_t))
    """
    def __init__(self, beta: float = 0.3, omega: float = 0.8, eps: float = 1e-8):
        assert 0 <= beta <= 1 and 0 <= omega <= 1 and beta < omega
        self.beta = beta
        self.omega = omega
        self.eps = eps

    def sample_mask_rate(self, batch_shape, device):
        u = torch.rand(batch_shape, device=device)
        return self.beta + (self.omega - self.beta) * u

    def loss_weight(self, mask_rate):
        return (self.omega - self.beta) / (mask_rate + self.eps)

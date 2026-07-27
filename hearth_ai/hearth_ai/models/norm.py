import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square LayerNorm (used in Llama, PaLM, etc.).
    Cheaper than LayerNorm since it skips mean-centering, and works
    fine as a drop-in replacement for the encoder blocks."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x * self.weight

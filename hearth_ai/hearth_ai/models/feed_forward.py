import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import HearthConfig


class SwiGLUFeedForward(nn.Module):
    """SwiGLU FFN, as used in Llama/PaLM. Gate and up projections are
    computed separately, gated with SiLU, then projected back down."""

    def __init__(self, config: HearthConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.ffn_hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.ffn_hidden_size, bias=False)
        self.down_proj = nn.Linear(config.ffn_hidden_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated = F.silu(self.gate_proj(x)) * self.up_proj(x)
        return self.dropout(self.down_proj(gated))

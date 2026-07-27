import torch
import torch.nn as nn

from ..config import HearthConfig
from .attention import MultiHeadSelfAttention
from .feed_forward import SwiGLUFeedForward
from .norm import RMSNorm


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: x -> x + Attn(RMSNorm(x)) -> x + FFN(RMSNorm(x))."""

    def __init__(self, config: HearthConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention = MultiHeadSelfAttention(config)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.feed_forward = SwiGLUFeedForward(config)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None):
        x = x + self.attention(self.attn_norm(x), attention_mask)
        x = x + self.feed_forward(self.ffn_norm(x))
        return x

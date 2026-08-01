import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import HearthConfig


class RotaryEmbedding(nn.Module):
    """Rotary positional embeddings (RoPE). Precomputes cos/sin tables and
    rotates query/key vectors so attention naturally encodes relative
    position, without a separate additive positional embedding."""

    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10_000.0):
        super().__init__()
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2).float() / head_dim)
        )
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)  # (max_seq_len, head_dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (max_seq_len, head_dim)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int, device, dtype):
        return (
            self.cos_cached[:seq_len].to(device=device, dtype=dtype),
            self.sin_cached[:seq_len].to(device=device, dtype=dtype),
        )


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # q, k: (B, num_heads, T, head_dim); cos/sin: (T, head_dim)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_rot = (q * cos) + (_rotate_half(q) * sin)
    k_rot = (k * cos) + (_rotate_half(k) * sin)
    return q_rot, k_rot


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config: HearthConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        self.rope = RotaryEmbedding(
            config.head_dim, config.max_seq_len, config.rope_theta
        )
        self.dropout = config.dropout

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None):
        # x: (B, T, H); attention_mask: (B, T) with 1 = keep, 0 = pad
        B, T, H = x.shape

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(T, x.device, x.dtype)
        q, k = apply_rope(q, k, cos, sin)

        attn_bias = None
        if attention_mask is not None:
            # (B, 1, 1, T) additive mask: 0 for keep, -inf for pad
            attn_bias = (1.0 - attention_mask[:, None, None, :].to(x.dtype)) * torch.finfo(x.dtype).min

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_bias,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, H)
        return self.out_proj(out)

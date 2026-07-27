import torch
import torch.nn as nn

from ..config import HearthConfig


class TokenEmbedding(nn.Module):
    """Token embedding. Positional information is injected separately via
    RoPE inside the attention layer, so no additive positional embedding
    is needed here (this is the modern Llama/RoPE-style approach the doc
    mentioned as an alternative to learned positional embeddings)."""

    def __init__(self, config: HearthConfig):
        super().__init__()
        self.embedding = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.dropout = nn.Dropout(config.dropout)
        self.scale = config.hidden_size ** 0.5

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids) * self.scale
        return self.dropout(x)

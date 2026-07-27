import torch
import torch.nn as nn

from ..config import HearthConfig, assert_within_param_budget, count_parameters
from .embeddings import TokenEmbedding
from .norm import RMSNorm
from .transformer_block import TransformerBlock


class HearthEncoder(nn.Module):
    """Shared backbone: Embedding -> N x TransformerBlock -> final RMSNorm.

    Default ``HearthConfig()`` is ~90M params. Construction asserts the module
    stays ≤ 200M (``MAX_MODEL_PARAMS``).

    Produces per-token hidden states plus a pooled "cognitive state" vector
    that every task head consumes. One encoder instance's weights can be
    shared/copied across as many HearthModel(head=...) wrappers as you like,
    or each task can start from its own copy and fine-tune independently -
    your choice at training time.
    """

    def __init__(self, config: HearthConfig, *, enforce_budget: bool = True):
        super().__init__()
        self.config = config
        self.embedding = TokenEmbedding(config)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_layers)]
        )
        self.final_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.apply(self._init_weights)
        if enforce_budget:
            assert_within_param_budget(self, name="HearthEncoder")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        """
        input_ids: (B, T) long tensor of token ids
        attention_mask: (B, T) with 1 = real token, 0 = padding

        Returns:
            hidden_states: (B, T, H) - per-token representations
            pooled: (B, H) - mean-pooled "cognitive state" over real tokens,
                     the input every task head expects.
        """
        x = self.embedding(input_ids)
        for block in self.blocks:
            x = block(x, attention_mask)
        hidden_states = self.final_norm(x)

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)  # (B, T, 1)
            summed = (hidden_states * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            pooled = summed / counts
        else:
            pooled = hidden_states.mean(dim=1)

        return hidden_states, pooled

    def num_parameters(self) -> int:
        return count_parameters(self)

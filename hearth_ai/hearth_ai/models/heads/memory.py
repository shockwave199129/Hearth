import torch.nn as nn

from ...labels import memory_num_types


class MemoryHead(nn.Module):
    """Decides whether to store a message as long-term memory, what kind,
    and how important it is. Three small sub-heads share one trunk:

      store_logit      (B, 1)            -> BCEWithLogitsLoss  (store / don't)
      type_logits       (B, num_types)   -> CrossEntropyLoss   (memory_type)
      importance_logit  (B, 1)           -> BCEWithLogitsLoss or MSE on
                                             a 0-1 target (importance score)

    Types locked in hearth_ai/labels/memory.yaml (8).
    Train all three jointly with MemoryLoss.
    """

    def __init__(
        self,
        hidden_size: int,
        num_memory_types: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        num_memory_types = num_memory_types or memory_num_types()
        trunk_size = hidden_size // 2
        self.trunk = nn.Sequential(
            nn.Linear(hidden_size, trunk_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.store_head = nn.Linear(trunk_size, 1)
        self.type_head = nn.Linear(trunk_size, num_memory_types)
        self.importance_head = nn.Linear(trunk_size, 1)

    def forward(self, pooled):
        h = self.trunk(pooled)
        return {
            "store_logit": self.store_head(h).squeeze(-1),
            "type_logits": self.type_head(h),
            "importance_logit": self.importance_head(h).squeeze(-1),
        }

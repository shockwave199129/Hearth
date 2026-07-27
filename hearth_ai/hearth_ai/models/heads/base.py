import torch.nn as nn


class HearthHead(nn.Module):
    """Base class for lightweight task heads that sit on top of the shared
    HearthEncoder's pooled output. Subclasses just need to set self.net
    and, if useful, self.output_names for readability."""

    def __init__(self, hidden_size: int, output_size: int, dropout: float = 0.1):
        super().__init__()
        self.output_size = output_size
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size),
        )

    def forward(self, pooled):
        return self.net(pooled)

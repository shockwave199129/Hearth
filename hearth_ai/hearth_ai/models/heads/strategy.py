from ...labels import strategy_num_labels
from .base import HearthHead


class StrategyHead(HearthHead):
    """Suggested response strategy (single-label CE).

    Locked 12 labels in hearth_ai/labels/strategy.yaml.
    PromptBuilder context only — Scheduler owns strategy (Book invariant).
    """

    def __init__(
        self,
        hidden_size: int,
        num_strategies: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__(hidden_size, num_strategies or strategy_num_labels(), dropout)

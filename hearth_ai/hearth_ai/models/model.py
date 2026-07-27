import torch.nn as nn

from ..config import HearthConfig, assert_within_param_budget, count_parameters
from .encoder import HearthEncoder
from .heads import (
    EmotionHead,
    IntentHead,
    MemoryHead,
    RelationshipHead,
    StrategyHead,
)


class HearthModel(nn.Module):
    """Single-task model: shared encoder + one head.

    Usage:
        encoder = HearthEncoder(config)
        emotion_model = HearthModel(encoder, EmotionHead(config.hidden_size))
        intent_model = HearthModel(encoder, IntentHead(config.hidden_size))

    Pass the *same* encoder instance to multiple HearthModel wrappers if you
    want them to share weights (multi-task learning, updated by whichever
    task's batch is currently training). Pass separate HearthEncoder(config)
    instances if you'd rather fine-tune each task independently.

    Construction asserts total params ≤ 200M.
    """

    def __init__(
        self,
        encoder: HearthEncoder,
        head: nn.Module,
        *,
        enforce_budget: bool = True,
    ):
        super().__init__()
        self.encoder = encoder
        self.head = head
        if enforce_budget:
            assert_within_param_budget(self, name="HearthModel")

    def forward(self, input_ids, attention_mask=None):
        _, pooled = self.encoder(input_ids, attention_mask)
        return self.head(pooled)

    def num_parameters(self) -> int:
        return count_parameters(self)


class HearthMultiTaskModel(nn.Module):
    """One shared encoder feeding several heads at once. Forward pass runs
    every head; the trainer decides which head's loss to backprop for a
    given batch (see trainer/train.py: MultiTaskTrainer).

    Construction asserts total params ≤ 200M.
    """

    def __init__(
        self,
        encoder: HearthEncoder,
        heads: dict,
        *,
        enforce_budget: bool = True,
    ):
        """
        heads: dict mapping task name -> head module, e.g.
            {"emotion": EmotionHead(...), "intent": IntentHead(...), ...}
        """
        super().__init__()
        self.encoder = encoder
        self.heads = nn.ModuleDict(heads)
        if enforce_budget:
            assert_within_param_budget(self, name="HearthMultiTaskModel")

    def forward(self, input_ids, attention_mask=None, tasks=None):
        """tasks: optional list of task names to run; runs all heads if None."""
        _, pooled = self.encoder(input_ids, attention_mask)
        tasks = tasks or list(self.heads.keys())
        return {task: self.heads[task](pooled) for task in tasks}

    def num_parameters(self) -> int:
        return count_parameters(self)


def build_encoder(config: HearthConfig | None = None) -> HearthEncoder:
    """Build a budget-checked encoder (default config ≈ 90M params)."""
    return HearthEncoder(config or HearthConfig())


def build_model(
    head: nn.Module,
    config: HearthConfig | None = None,
    encoder: HearthEncoder | None = None,
) -> HearthModel:
    """Build encoder+head with ≤200M param assert."""
    enc = encoder or build_encoder(config)
    return HearthModel(enc, head)


def build_multitask_model(
    config: HearthConfig | None = None,
    encoder: HearthEncoder | None = None,
    *,
    include: tuple[str, ...] = (
        "emotion",
        "intent",
        "memory",
        "relationship",
        "strategy",
    ),
) -> HearthMultiTaskModel:
    """Build shared-encoder model with the five locked heads (≤200M)."""
    cfg = config or HearthConfig()
    enc = encoder or build_encoder(cfg)
    h = cfg.hidden_size
    registry = {
        "emotion": EmotionHead(h),
        "intent": IntentHead(h),
        "memory": MemoryHead(h),
        "relationship": RelationshipHead(h),
        "strategy": StrategyHead(h),
    }
    unknown = set(include) - set(registry)
    if unknown:
        raise ValueError(f"Unknown heads: {sorted(unknown)}")
    heads = {name: registry[name] for name in include}
    return HearthMultiTaskModel(enc, heads)

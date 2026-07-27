"""Ready-made loss functions matching each task head's output shape, so you
don't have to re-derive them per task. Pass one of these as `loss_fn` to
Trainer(...)."""
import torch.nn as nn


emotion_loss = nn.BCEWithLogitsLoss()      # EmotionHead: multi-label
intent_loss = nn.CrossEntropyLoss()        # IntentHead: single-label
relationship_loss = nn.SmoothL1Loss()      # RelationshipHead: regression
strategy_loss = nn.CrossEntropyLoss()      # StrategyHead: single-label


class MemoryLoss(nn.Module):
    """Combined loss for MemoryHead's three outputs. `label` must be a dict
    with the same keys the head returns: store (float 0/1), type (long class
    index), importance (float 0-1). Weight the three terms with
    `store_weight` / `type_weight` / `importance_weight` if one task matters
    more than the others for your use case."""

    def __init__(self, store_weight=1.0, type_weight=1.0, importance_weight=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.ce = nn.CrossEntropyLoss()
        self.store_weight = store_weight
        self.type_weight = type_weight
        self.importance_weight = importance_weight

    def forward(self, logits: dict, label: dict):
        store_loss = self.bce(logits["store_logit"], label["store"].float())
        type_loss = self.ce(logits["type_logits"], label["type"].long())
        importance_loss = self.bce(logits["importance_logit"], label["importance"].float())
        return (
            self.store_weight * store_loss
            + self.type_weight * type_loss
            + self.importance_weight * importance_loss
        )


memory_loss = MemoryLoss()

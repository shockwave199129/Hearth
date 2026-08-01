from ...labels import emotion_num_labels
from .base import HearthHead


class EmotionHead(HearthHead):
    """Multi-label emotion scores (GoEmotions 28 — see hearth_ai/labels/emotion.yaml).
    Train with nn.BCEWithLogitsLoss since a message can carry more than
    one emotion at once."""

    def __init__(
        self,
        hidden_size: int,
        num_emotions: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__(hidden_size, num_emotions or emotion_num_labels(), dropout)

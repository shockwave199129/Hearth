from ...labels import intent_num_labels
from .base import HearthHead


class IntentHead(HearthHead):
    """Primary companion need / intent (single-label CE).

    Locked labels (10): vent, validate, comfort, celebrate, advise, inquire,
    plan, small_talk, meta, unknown — see hearth_ai/labels/intent.yaml.
    """

    def __init__(
        self,
        hidden_size: int,
        num_intents: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__(hidden_size, num_intents or intent_num_labels(), dropout)

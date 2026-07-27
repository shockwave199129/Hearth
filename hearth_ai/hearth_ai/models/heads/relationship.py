from ...labels import relationship_num_signals
from .base import HearthHead


class RelationshipHead(HearthHead):
    """Continuous relationship signals (SmoothL1Loss).

    Locked order: trust_delta, vulnerability, openness, comfort —
    see hearth_ai/labels/relationship.yaml.
    Apply torch.tanh on trust_delta at inference if targets are bipolar.
    """

    def __init__(
        self,
        hidden_size: int,
        num_signals: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__(hidden_size, num_signals or relationship_num_signals(), dropout)

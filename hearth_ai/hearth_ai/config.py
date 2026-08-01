from dataclasses import asdict, dataclass
from typing import Any

# Hard cap from the NLP plan: each built model must stay ≤ 200M parameters.
# Default HearthConfig() lands around ~90M for the encoder alone (see estimate_param_count).
MAX_MODEL_PARAMS = 200_000_000


@dataclass
class HearthConfig:
    """Hyperparameters for HearthEncoder-Base (v1).

    Default sizing (~90M encoder params, well under the 200M budget):
      vocab=32k, hidden=512, layers=16, heads=8, head_dim=64, ffn=2048
      RoPE + RMSNorm + SwiGLU (encoder-only).

    Shrink for local smoke tests, e.g.::
        HearthConfig(hidden_size=128, num_layers=4, num_heads=4, head_dim=32,
                     ffn_hidden_size=256, vocab_size=2000)
    """

    vocab_size: int = 32_000
    max_seq_len: int = 512

    hidden_size: int = 512
    num_layers: int = 16
    num_heads: int = 8
    head_dim: int = 64  # hidden_size / num_heads

    ffn_hidden_size: int = 2048
    dropout: float = 0.1

    rope_theta: float = 10_000.0
    rms_norm_eps: float = 1e-6

    pad_token_id: int = 0

    # Soft documentation target for the default encoder (not a hard gate).
    target_encoder_params: int = 90_000_000

    def __post_init__(self):
        assert self.hidden_size == self.num_heads * self.head_dim, (
            f"hidden_size ({self.hidden_size}) must equal "
            f"num_heads * head_dim ({self.num_heads} * {self.head_dim})"
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HearthConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})

    def estimate_param_count(self) -> int:
        """Rough parameter estimate for HearthEncoder (no heads).

        Counts embeddings + per-layer attention/FFN/norms + final RMSNorm.
        Useful before allocating a full model; builders still assert on the
        real ``num_parameters()`` after construction.
        """
        h = self.hidden_size
        f = self.ffn_hidden_size
        # token embeddings
        n = self.vocab_size * h
        # attention: q,k,v,out projections (bias-free in our attn) ≈ 4 * h * h
        # SwiGLU: w1 (h->f), w2 (h->f), w3 (f->h) ≈ 2*h*f + f*h
        # 2 x RMSNorm scale vectors per block + 1 final
        per_layer = (4 * h * h) + (3 * h * f) + (2 * h)
        n += self.num_layers * per_layer
        n += h  # final RMSNorm
        return int(n)


def count_parameters(module) -> int:
    return sum(p.numel() for p in module.parameters())


def assert_within_param_budget(
    module,
    *,
    max_params: int = MAX_MODEL_PARAMS,
    name: str = "model",
) -> int:
    """Raise AssertionError if ``module`` exceeds the ≤200M param budget."""
    n = count_parameters(module)
    assert n <= max_params, (
        f"{name} has {n:,} parameters ({n / 1e6:.1f}M), "
        f"which exceeds the hard budget of {max_params:,} ({max_params / 1e6:.0f}M). "
        f"Shrink HearthConfig (layers/hidden/ffn/vocab) or drop heads."
    )
    return n

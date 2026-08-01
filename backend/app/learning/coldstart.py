"""The Cold-Start Problem (Book Vol 7 Ch 11) — one general,
confidence-weighted blend of population defaults and priors, applied by
every specific computation in this package (traits, affinity, trust)
rather than reimplemented per computation.

Population defaults are deliberately conservative: trust has no prior at
all (Volume 3's genuine-zero requirement); traits/affinity have only a
neutral, non-committal default absent a specific prior."""
from __future__ import annotations

DEFAULT_K = 5.0


def confidence_curve(sample_size: int, k: float = DEFAULT_K) -> float:
    """Smoothly increases 0 -> 1 as `sample_size` grows, rather than
    switching abruptly at a fixed threshold (Vol 7 Ch 11) — avoids a
    jarring change in behavior right at a cutoff point."""
    if sample_size <= 0:
        return 0.0
    return sample_size / (sample_size + k)


def blend(
    population_default: float,
    prior: float | None,
    observed: float,
    sample_size: int,
    *,
    k: float = DEFAULT_K,
) -> float:
    """`observed` is the standard-pipeline output computed from actual
    observations; `prior` (if any) is a related, better-grounded starting
    estimate (e.g. Ch 6's cross-affinity prior from Communication Traits);
    `population_default` is the conservative fallback when neither applies.
    Weight given to `observed` grows smoothly with `sample_size`."""
    confidence = confidence_curve(sample_size, k)
    baseline = prior if prior is not None else population_default
    return round(confidence * observed + (1 - confidence) * baseline, 4)

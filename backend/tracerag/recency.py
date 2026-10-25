"""Age-based score decay for Graphsight/TraceRAG nodes:
score *= 0.5 ** (age_days / half_life), bounded by RECENCY_FLOOR.
"""

from __future__ import annotations

import time
from . import config

SECONDS_PER_DAY = 86400.0


def age_days(ts: int | float | None, now: float | None = None) -> float | None:
    """Calculates age in days from UNIX timestamp, or None if no timestamp exists."""
    if not ts:
        return None
    current_time = time.time() if now is None else now
    return max(0.0, (current_time - float(ts)) / SECONDS_PER_DAY)


def half_life_for(node_type: str | None) -> float:
    """Retrieves half life in days tailored to node entity type."""
    return config.RECENCY_HALF_LIFE_DAYS.get(
        node_type or "", config.DEFAULT_HALF_LIFE_DAYS
    )


def decay_factor(
    ts: int | float | None,
    node_type: str | None = None,
    now: float | None = None
) -> float:
    """Multiplier in [RECENCY_FLOOR, 1.0]. 1.0 when recency is off or ts is absent."""
    if not config.RECENCY_ENABLED:
        return 1.0
    age = age_days(ts, now)
    if age is None:
        return 1.0
    factor = 0.5 ** (age / half_life_for(node_type))
    return max(config.RECENCY_FLOOR, factor)

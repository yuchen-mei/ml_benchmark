from __future__ import annotations

import math
from collections.abc import Sequence


def percentile(values: Sequence[float], percent: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else math.nan


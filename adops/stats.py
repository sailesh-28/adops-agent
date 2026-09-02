"""Statistical sufficiency and significance. Code decides what counts as evidence."""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

MIN_TRIALS = 60          # below this, a rate estimate is not usable
MIN_SUCCESSES = 5        # below this, a conversion-rate estimate is not usable
ALPHA = 0.05


def normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def two_proportion_test(x1: float, n1: float, x2: float, n2: float) -> Dict[str, object]:
    """Two-sided z-test for a difference in rates (e.g. CVR of A vs B).

    Returns a verdict of insufficient_data / not_significant / significant.
    """
    if n1 < MIN_TRIALS or n2 < MIN_TRIALS or (x1 + x2) < MIN_SUCCESSES:
        return {"verdict": "insufficient_data", "p_value": None, "z": None,
                "n1": n1, "n2": n2, "x1": x1, "x2": x2,
                "note": "sample too small for a reliable comparison"}
    p1, p2 = x1 / n1, x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    se = math.sqrt(max(p_pool * (1.0 - p_pool) * (1.0 / n1 + 1.0 / n2), 1e-12))
    z = (p1 - p2) / se
    p_value = 2.0 * (1.0 - normal_cdf(abs(z)))
    return {
        "verdict": "significant" if p_value < ALPHA else "not_significant",
        "p_value": round(p_value, 4), "z": round(z, 2),
        "n1": n1, "n2": n2, "x1": x1, "x2": x2,
        "rate1": p1, "rate2": p2,
    }


def wilson_interval(x: float, n: float, z: float = 1.96) -> Optional[Sequence[float]]:
    """Wilson score interval for a rate -- honest uncertainty at small n."""
    if n <= 0:
        return None
    p = x / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = (z * math.sqrt(max(p * (1 - p) / n + z * z / (4 * n * n), 0.0))) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def linear_trend(values: Sequence[Optional[float]]) -> Dict[str, object]:
    """OLS slope over an evenly spaced series, expressed per-period and as a
    fraction of the series mean. Ignores None entries."""
    pts = [(i, v) for i, v in enumerate(values) if v is not None]
    if len(pts) < 3:
        return {"slope": None, "slope_pct_per_period": None, "n": len(pts)}
    n = float(len(pts))
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    denom = sum((p[0] - mx) ** 2 for p in pts)
    if denom == 0:
        return {"slope": None, "slope_pct_per_period": None, "n": int(n)}
    slope = sum((p[0] - mx) * (p[1] - my) for p in pts) / denom
    return {
        "slope": slope,
        "slope_pct_per_period": (slope / my) if my else None,
        "mean": my,
        "n": int(n),
    }


def sufficiency(clicks: float, conversions: float) -> str:
    """Is there enough volume to judge this entity's conversion performance?"""
    if clicks < MIN_TRIALS or conversions < MIN_SUCCESSES:
        return "insufficient"
    if clicks < 3 * MIN_TRIALS or conversions < 3 * MIN_SUCCESSES:
        return "thin"
    return "adequate"

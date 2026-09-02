"""All derived-metric arithmetic. The LLM never computes any of this."""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

RAW_FIELDS = ["spend", "impressions", "reach", "clicks", "conversions", "revenue"]


def _div(num: float, den: float) -> Optional[float]:
    """Safe division. Returns None when undefined -- never 0, never inf."""
    if den is None or den == 0:
        return None
    return num / den


def derive(raw: Dict[str, float]) -> Dict[str, Optional[float]]:
    """Compute standard performance metrics from raw counts."""
    spend = float(raw.get("spend", 0.0) or 0.0)
    impr = float(raw.get("impressions", 0) or 0)
    reach = float(raw.get("reach", 0) or 0)
    clicks = float(raw.get("clicks", 0) or 0)
    conv = float(raw.get("conversions", 0.0) or 0.0)
    rev = float(raw.get("revenue", 0.0) or 0.0)
    return {
        "ctr": _div(clicks, impr),
        "cpc": _div(spend, clicks),
        "cpm": _div(spend * 1000.0, impr),
        "cvr": _div(conv, clicks),
        "cpa": _div(spend, conv),
        "roas": _div(rev, spend),
        "frequency": _div(impr, reach),
        "aov": _div(rev, conv),
    }


def zero_raw() -> Dict[str, float]:
    return {f: 0.0 for f in RAW_FIELDS}


def sum_raw(rows: Iterable[dict]) -> Dict[str, float]:
    """Aggregate raw counts. Only additive fields are summed -- never ratios."""
    tot = zero_raw()
    for r in rows:
        for f in RAW_FIELDS:
            tot[f] += float(r.get(f, 0) or 0)
    return tot


def aggregate(rows: Iterable[dict]) -> Dict[str, Optional[float]]:
    """Sum raw counts, then derive. (Never average a ratio of ratios.)"""
    tot = sum_raw(rows)
    out: Dict[str, Optional[float]] = dict(tot)
    out.update(derive(tot))
    return out


def rows_for(rows: Iterable[dict], **filters) -> List[dict]:
    out = []
    for r in rows:
        if all(r.get(k) == v for k, v in filters.items()):
            out.append(r)
    return out


def pct_change(new: Optional[float], old: Optional[float]) -> Optional[float]:
    """Relative change as a fraction. None when undefined."""
    if new is None or old is None or old == 0:
        return None
    return (new - old) / old


def fmt(value: Optional[float], kind: str = "num") -> str:
    """Deterministic formatting for prompts and reports."""
    if value is None:
        return "n/a"
    if kind == "money":
        return "%.2f" % value
    if kind == "pct":
        return "%.2f%%" % (value * 100.0)
    if kind == "pct_delta":
        return "%+.1f%%" % (value * 100.0)
    if kind == "int":
        return "%d" % int(round(value))
    return "%.3f" % value

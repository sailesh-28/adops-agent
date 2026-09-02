"""ORIENT step: a deterministic, DESCRIPTIVE evidence briefing.

Deliberately contains no causal labels ("fatigue", "saturation") and no
recommendations. It reports measurements, deltas, sample sizes and statistical
sufficiency only -- diagnosis is the agent's job, not the briefing's.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from . import metrics as M
from . import stats as S


def constraint_status(env) -> dict:
    obj = env.objective
    spend = env.spend_to_date()
    rows = env.live_rows()
    agg = M.aggregate(rows)
    cpa = agg["cpa"]
    elapsed = max(0, env.live_period)
    planned = obj.planned_weekly_spend() * max(1, elapsed)
    account_daily = env.account.account_daily_budget()
    projected = spend + account_daily * 7.0 * env.periods_remaining()
    return {
        "periods_elapsed": elapsed,
        "periods_remaining": env.periods_remaining(),
        "target_cpa": obj.target_cpa,
        "cpa_to_date": round(cpa, 2) if cpa else None,
        "cpa_vs_target_pct": (round((cpa / obj.target_cpa - 1) * 100, 1)
                              if cpa else None),
        "conversions_to_date": round(agg["conversions"] or 0.0, 1),
        "spend_to_date": round(spend, 2),
        "total_budget": obj.total_budget,
        "budget_remaining": round(obj.total_budget - spend, 2),
        "pacing_vs_plan_pct": (round((spend / planned - 1) * 100, 1)
                               if elapsed and planned else None),
        "account_daily_budget": round(account_daily, 2),
        "account_daily_budget_ceiling": env.limits.max_account_daily_budget,
        "projected_total_spend_if_unchanged": round(projected, 2),
        "projected_within_budget": projected <= obj.total_budget + 1e-6,
    }


def budget_status(env) -> dict:
    L = env.limits
    return {
        "structural_changes_used": env.changes_total,
        "structural_changes_max": L.max_total_changes,
        "changes_used_this_period": env.changes_this_period,
        "changes_max_this_period": L.max_changes_per_period,
        "tool_calls_this_period": env.tool_calls_this_period,
        "tool_calls_max_this_period": L.max_tool_calls_per_period,
        "creative_tests_used": env.creative_test_count,
        "creative_tests_max": L.max_active_creative_tests,
    }


def _window(env, back: int) -> List[int]:
    return [max(1, env.period - back + 1), env.period]


def ad_set_table(env, lookback: int = 3) -> List[dict]:
    """One descriptive row per ad set: current window, previous window, and deltas."""
    cur_a, cur_b = _window(env, lookback)
    prev_b = cur_a - 1
    prev_a = max(1, prev_b - lookback + 1)
    out = []
    for aid, ad_set in env.account.ad_sets.items():
        cur = env.entity_metrics(aid, cur_a, cur_b)
        prev = env.entity_metrics(aid, prev_a, prev_b) if prev_b >= 1 else {}
        row = {
            "ad_set_id": aid,
            "name": ad_set.name,
            "campaign_id": ad_set.campaign_id,
            "status": ad_set.status,
            "daily_budget": ad_set.daily_budget,
            "active_creatives": len(env.account.active_creatives(aid)),
            "spend": round(cur["spend"], 2),
            "clicks": int(cur["clicks"]),
            "conversions": round(cur["conversions"], 1),
            "cpa": round(cur["cpa"], 2) if cur["cpa"] else None,
            "cpc": round(cur["cpc"], 2) if cur["cpc"] else None,
            "cpm": round(cur["cpm"], 2) if cur["cpm"] else None,
            "ctr_pct": round(cur["ctr"] * 100, 3) if cur["ctr"] else None,
            "cvr_pct": round(cur["cvr"] * 100, 3) if cur["cvr"] else None,
            "reach": int(cur["reach"]),
            "frequency": round(cur["frequency"], 2) if cur["frequency"] else None,
            "roas": round(cur["roas"], 2) if cur["roas"] else None,
            "data_sufficiency": S.sufficiency(cur["clicks"], cur["conversions"]),
        }
        for m in ("cpa", "ctr", "cvr", "cpm", "frequency"):
            row["%s_delta_pct" % m] = (
                round(M.pct_change(cur.get(m), prev.get(m)) * 100, 1)
                if prev and M.pct_change(cur.get(m), prev.get(m)) is not None else None)
        out.append(row)
    out.sort(key=lambda r: (r["cpa"] is None, -(r["cpa"] or 0)))
    return out


def account_summary(env, lookback: int = 3) -> dict:
    a, b = _window(env, lookback)
    cur = env.entity_metrics_all(a, b)
    prev_b = a - 1
    prev = env.entity_metrics_all(max(1, prev_b - lookback + 1), prev_b) if prev_b >= 1 else {}
    out = {
        "window_periods": [a, b],
        "spend": round(cur["spend"], 2),
        "conversions": round(cur["conversions"], 1),
        "cpa": round(cur["cpa"], 2) if cur["cpa"] else None,
        "ctr_pct": round(cur["ctr"] * 100, 3) if cur["ctr"] else None,
        "cvr_pct": round(cur["cvr"] * 100, 3) if cur["cvr"] else None,
        "cpm": round(cur["cpm"], 2) if cur["cpm"] else None,
        "roas": round(cur["roas"], 2) if cur["roas"] else None,
    }
    if prev:
        for m in ("cpa", "ctr", "cvr", "cpm"):
            ch = M.pct_change(cur.get(m), prev.get(m))
            out["%s_delta_pct" % m] = round(ch * 100, 1) if ch is not None else None
    return out


def ledger_view(env) -> List[dict]:
    return [{
        "seq": e.seq, "period": e.period, "tool": e.tool, "args": e.args,
        "rationale": e.rationale, "expected": e.expected,
        "verdict": e.verdict, "detail": e.detail,
    } for e in env.ledger]


def build(env, lookback: int = 3) -> dict:
    return {
        "period": env.live_period,
        "horizon": env.objective.horizon_periods,
        "objective": {
            "description": env.objective.description,
            "target_cpa": env.objective.target_cpa,
            "total_budget": env.objective.total_budget,
            "currency": env.objective.currency,
            "horizon_periods": env.objective.horizon_periods,
        },
        "constraints": constraint_status(env),
        "budgets": budget_status(env),
        "account": account_summary(env, lookback),
        "ad_sets": ad_set_table(env, lookback),
        "action_ledger": ledger_view(env),
        "reporting_note": ("Conversions are reported with a delay: roughly a quarter of a "
                           "period's conversions appear in the following period's figures."),
    }

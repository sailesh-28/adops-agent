"""Tool layer: the only surface through which a policy may touch the environment.

Every call is validated by guardrails.check() first. Results use a uniform envelope
so a rejection is just another observation the agent must react to.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

from . import brief as B
from . import metrics as M
from . import stats as S
from .guardrails import Rejection, check

# ------------------------------------------------------------------ schemas
TOOL_SCHEMAS: List[dict] = [
    {
        "name": "get_account_snapshot",
        "kind": "read",
        "description": ("Current structure and performance of every campaign, ad set and "
                        "creative over the last `lookback_periods` periods, with "
                        "period-over-period deltas and data-sufficiency verdicts."),
        "args": {"lookback_periods": "int, 1-6, default 3"},
    },
    {
        "name": "get_timeseries",
        "kind": "read",
        "description": ("Per-period history for one entity (campaign, ad set or creative) "
                        "plus fitted trends for CTR, CVR, CPA, CPM and frequency."),
        "args": {"entity_id": "string", "periods": "int, 3-12, default 8"},
    },
    {
        "name": "compare_segments",
        "kind": "read",
        "description": ("Compare sibling entities on a chosen dimension. Returns each "
                        "sibling's metrics plus a statistical test of its CVR and CTR "
                        "against the pooled rest of the group. Use this before acting on "
                        "an apparent difference: it tells you whether the difference is "
                        "distinguishable from noise at the available sample size."),
        "args": {"level": "'ad_set' or 'creative'",
                 "parent_id": "campaign id (for level=ad_set) or ad_set id (for level=creative); "
                              "omit for all ad sets in the account",
                 "lookback_periods": "int, 1-6, default 3"},
    },
    {
        "name": "set_budget",
        "kind": "action",
        "description": ("Change one ad set's daily budget. Limited to a "
                        "single-step change; a change of 20% or more puts the ad set into "
                        "a learning phase with reduced efficiency for the next period."),
        "args": {"entity_id": "ad_set id", "new_daily_budget": "number",
                 "rationale": "string", "expected_effect": "object (see below)"},
    },
    {
        "name": "set_status",
        "kind": "action",
        "description": ("Pause or resume an ad set or a creative. Always triggers a "
                        "learning phase on the affected ad set."),
        "args": {"entity_id": "ad_set or creative id", "status": "'active' or 'paused'",
                 "rationale": "string", "expected_effect": "object"},
    },
    {
        "name": "launch_creative_test",
        "kind": "action",
        "description": ("Add one new creative to an ad set. Its quality is unknown in "
                        "advance and may be better or worse than the existing creatives. "
                        "Triggers a learning phase."),
        "args": {"ad_set_id": "ad_set id", "rationale": "string",
                 "expected_effect": "object"},
    },
    {
        "name": "advance_period",
        "kind": "control",
        "description": ("Let one period of performance elapse and observe the new state. "
                        "This is also how you WAIT: advancing without acting is a valid "
                        "and sometimes correct decision."),
        "args": {"note": "string, why you are advancing now"},
    },
    {
        "name": "finish",
        "kind": "control",
        "description": "End the run early, when no further action has positive expected value.",
        "args": {"reason": "string", "summary": "string"},
    },
]

EXPECTED_EFFECT_SPEC = {
    "entity_id": "the entity whose metric you expect to move",
    "metric": "one of cpa, ctr, cvr, cpc, cpm, conversions, spend, roas, frequency",
    "direction": "'increase' or 'decrease'",
    "magnitude_pct": "number 1-300: how large a move you expect, in percent",
    "periods": "integer 1-4: within how many periods the effect should be visible",
}

CONSIDERED_ALTERNATIVE_SPEC = {
    "level": "'creative' or 'ad_set' -- the OTHER hierarchy level from the one you "
             "are about to act on",
    "summary": "the specific alternative action at that level (e.g. 'pause the "
               "weaker creative in this ad set' if you are about to cut the ad "
               "set's budget)",
    "why_rejected": "why the evidence favours your chosen action over this "
                    "alternative; if no plausible alternative exists, say so and why",
}

PATTERN_CHECK_SPEC = {
    "similar_past_action": "the seq number of a past action of the same shape (same "
                           "tool, same kind of entity, same direction of change), or "
                           "'none' if there is no such precedent",
    "reasoning": "if there is a precedent, state its measured verdict and why this "
                "situation is different, or why acting again is still justified; if "
                "there is no precedent, say so",
}


def envelope(ok: bool, tool: str, data=None, error: Optional[str] = None,
             code: Optional[str] = None) -> dict:
    out = {"ok": ok, "tool": tool}
    if ok:
        out["data"] = data
    else:
        out["error"] = error
        out["error_code"] = code
    return out


# -------------------------------------------------------------- read tools
def _entity_window(env, lookback: int):
    lookback = max(1, min(6, int(lookback)))
    return max(1, env.period - lookback + 1), env.period


def t_get_account_snapshot(env, lookback_periods: int = 3, **_) -> dict:
    a, b = _entity_window(env, lookback_periods)
    campaigns = []
    for cid, camp in env.account.campaigns.items():
        m = env.entity_metrics(cid, a, b)
        campaigns.append({
            "campaign_id": cid, "name": camp.name, "status": camp.status,
            "daily_budget": round(sum(x.daily_budget for x in env.account.active_ad_sets(cid)), 2),
            "spend": round(m["spend"], 2), "conversions": round(m["conversions"], 1),
            "cpa": round(m["cpa"], 2) if m["cpa"] else None,
            "ad_set_ids": camp.ad_set_ids,
        })
    creatives = []
    for crid, cr in env.account.creatives.items():
        m = env.entity_metrics(crid, a, b)
        creatives.append({
            "creative_id": crid, "name": cr.name, "ad_set_id": cr.ad_set_id,
            "status": cr.status, "is_test": cr.is_test,
            "spend": round(m["spend"], 2), "clicks": int(m["clicks"]),
            "conversions": round(m["conversions"], 1),
            "ctr_pct": round(m["ctr"] * 100, 3) if m["ctr"] else None,
            "cvr_pct": round(m["cvr"] * 100, 3) if m["cvr"] else None,
            "cpa": round(m["cpa"], 2) if m["cpa"] else None,
            "cpc": round(m["cpc"], 2) if m["cpc"] else None,
            "data_sufficiency": S.sufficiency(m["clicks"], m["conversions"]),
        })
    return {
        "window_periods": [a, b],
        "account": B.account_summary(env, lookback_periods),
        "campaigns": campaigns,
        "ad_sets": B.ad_set_table(env, lookback_periods),
        "creatives": creatives,
    }


def t_get_timeseries(env, entity_id: str = "", periods: int = 8, **_) -> dict:
    kind = env.account.entity_type(entity_id)
    if kind is None:
        raise Rejection("unknown_entity", "No such entity: %r" % entity_id)
    periods = max(3, min(12, int(periods)))
    first = max(1, env.period - periods + 1)
    series = []
    for p in range(first, env.period + 1):
        m = env.entity_metrics(entity_id, p, p)
        series.append({
            "period": p,
            "relative_period": p - env.live_start_period,   # <=0 means historical
            "spend": round(m["spend"], 2),
            "impressions": int(m["impressions"]),
            "reach": int(m["reach"]),
            "clicks": int(m["clicks"]),
            "conversions": round(m["conversions"], 1),
            "ctr_pct": round(m["ctr"] * 100, 3) if m["ctr"] else None,
            "cvr_pct": round(m["cvr"] * 100, 3) if m["cvr"] else None,
            "cpc": round(m["cpc"], 2) if m["cpc"] else None,
            "cpm": round(m["cpm"], 2) if m["cpm"] else None,
            "cpa": round(m["cpa"], 2) if m["cpa"] else None,
            "frequency": round(m["frequency"], 2) if m["frequency"] else None,
        })
    trends = {}
    for key in ("ctr_pct", "cvr_pct", "cpa", "cpm", "frequency", "cpc"):
        tr = S.linear_trend([row[key] for row in series])
        trends[key] = {
            "slope_per_period": round(tr["slope"], 5) if tr["slope"] is not None else None,
            "pct_per_period": (round(tr["slope_pct_per_period"] * 100, 2)
                               if tr.get("slope_pct_per_period") is not None else None),
            "points": tr["n"],
        }
    return {"entity_id": entity_id, "entity_type": kind,
            "name": env.account.name_of(entity_id),
            "series": series, "trends": trends,
            "note": "relative_period <= 0 is history from before the current run."}


def t_compare_segments(env, level: str = "ad_set", parent_id: Optional[str] = None,
                       lookback_periods: int = 3, **_) -> dict:
    a, b = _entity_window(env, lookback_periods)
    if level == "ad_set":
        if parent_id:
            if env.account.entity_type(parent_id) != "campaign":
                raise Rejection("invalid_parent",
                                "level='ad_set' expects a campaign id as parent_id.")
            ids = env.account.campaigns[parent_id].ad_set_ids
        else:
            ids = list(env.account.ad_sets)
    elif level == "creative":
        if env.account.entity_type(parent_id) != "ad_set":
            raise Rejection("invalid_parent",
                            "level='creative' requires an ad_set id as parent_id.")
        ids = list(env.account.ad_sets[parent_id].creative_ids)
    else:
        raise Rejection("invalid_level", "level must be 'ad_set' or 'creative'.")

    raw = {i: M.sum_raw(env.entity_rows(i, a, b)) for i in ids}
    segments = []
    for i in ids:
        mine = raw[i]
        rest_clicks = sum(raw[j]["clicks"] for j in ids if j != i)
        rest_conv = sum(raw[j]["conversions"] for j in ids if j != i)
        rest_impr = sum(raw[j]["impressions"] for j in ids if j != i)
        m = M.aggregate([mine])
        cvr_test = S.two_proportion_test(mine["conversions"], mine["clicks"],
                                         rest_conv, rest_clicks)
        ctr_test = S.two_proportion_test(mine["clicks"], mine["impressions"],
                                         rest_clicks, rest_impr)
        ci = S.wilson_interval(mine["conversions"], mine["clicks"])
        segments.append({
            "entity_id": i,
            "name": env.account.name_of(i),
            "status": (env.account.ad_sets[i].status if level == "ad_set"
                       else env.account.creatives[i].status),
            "spend": round(mine["spend"], 2),
            "clicks": int(mine["clicks"]),
            "conversions": round(mine["conversions"], 1),
            "cpa": round(m["cpa"], 2) if m["cpa"] else None,
            "ctr_pct": round(m["ctr"] * 100, 3) if m["ctr"] else None,
            "cvr_pct": round(m["cvr"] * 100, 3) if m["cvr"] else None,
            "cvr_95ci_pct": [round(ci[0] * 100, 3), round(ci[1] * 100, 3)] if ci else None,
            "cpm": round(m["cpm"], 2) if m["cpm"] else None,
            "frequency": round(m["frequency"], 2) if m["frequency"] else None,
            "data_sufficiency": S.sufficiency(mine["clicks"], mine["conversions"]),
            "cvr_vs_rest": {"verdict": cvr_test["verdict"], "p_value": cvr_test.get("p_value")},
            "ctr_vs_rest": {"verdict": ctr_test["verdict"], "p_value": ctr_test.get("p_value")},
        })
    segments.sort(key=lambda r: (r["cpa"] is None, -(r["cpa"] or 0)))
    return {"level": level, "parent_id": parent_id, "window_periods": [a, b],
            "segments": segments,
            "note": ("Verdicts compare each segment against the pooled remainder of the "
                     "group. 'insufficient_data' means the sample cannot support a "
                     "conclusion, not that the segments are equal.")}


# ------------------------------------------------------------ action tools
def t_set_budget(env, entity_id: str = "", new_daily_budget: float = 0.0,
                 rationale: str = "", expected_effect: Optional[dict] = None,
                 considered_alternative: Optional[dict] = None,
                 pattern_check: Optional[dict] = None, **_) -> dict:
    return env.apply_set_budget(entity_id, new_daily_budget, rationale,
                                expected_effect or {}, considered_alternative or {},
                                pattern_check or {})


def t_set_status(env, entity_id: str = "", status: str = "", rationale: str = "",
                 expected_effect: Optional[dict] = None,
                 considered_alternative: Optional[dict] = None,
                 pattern_check: Optional[dict] = None, **_) -> dict:
    return env.apply_set_status(entity_id, status, rationale, expected_effect or {},
                                considered_alternative or {}, pattern_check or {})


def t_launch_creative_test(env, ad_set_id: str = "", rationale: str = "",
                           expected_effect: Optional[dict] = None,
                           considered_alternative: Optional[dict] = None,
                           pattern_check: Optional[dict] = None, **_) -> dict:
    return env.apply_launch_creative_test(ad_set_id, rationale, expected_effect or {},
                                          considered_alternative or {}, pattern_check or {})


def t_advance_period(env, note: str = "", **_) -> dict:
    res = env.advance()
    if not res.get("ok"):
        raise Rejection("horizon_reached", "The planning horizon is complete.")
    res["note_recorded"] = note
    res["resolved_effects"] = [
        {"seq": e.seq, "tool": e.tool, "verdict": e.verdict, "detail": e.detail}
        for e in env.ledger if e.verdict != "pending"
    ]
    return res


def t_finish(env, reason: str = "", summary: str = "", **_) -> dict:
    env.finished_reason = reason or "agent_finished"
    return {"finished": True, "reason": env.finished_reason, "summary": summary}


REGISTRY: Dict[str, Callable] = {
    "get_account_snapshot": t_get_account_snapshot,
    "get_timeseries": t_get_timeseries,
    "compare_segments": t_compare_segments,
    "set_budget": t_set_budget,
    "set_status": t_set_status,
    "launch_creative_test": t_launch_creative_test,
    "advance_period": t_advance_period,
    "finish": t_finish,
}


def dispatch(env, tool: str, args: Optional[dict] = None) -> dict:
    """Validate then execute. Never raises: a rejection becomes an observation."""
    args = dict(args or {})
    if tool not in REGISTRY:
        env.rejections.append({"period": env.live_period, "tool": tool,
                               "code": "unknown_tool", "args": args})
        return envelope(False, tool, error="No such tool: %r. Available: %s"
                        % (tool, ", ".join(sorted(REGISTRY))), code="unknown_tool")
    try:
        check(env, tool, args)
    except Rejection as r:
        env.rejections.append({"period": env.live_period, "tool": tool,
                               "code": r.code, "message": r.message, "args": args})
        env.tool_calls_total += 1
        return envelope(False, tool, error=r.message, code=r.code)
    try:
        data = REGISTRY[tool](env, **args)
    except Rejection as r:
        env.rejections.append({"period": env.live_period, "tool": tool,
                               "code": r.code, "message": r.message, "args": args})
        env.tool_calls_total += 1
        return envelope(False, tool, error=r.message, code=r.code)
    except TypeError as exc:
        env.rejections.append({"period": env.live_period, "tool": tool,
                               "code": "bad_arguments", "message": str(exc), "args": args})
        env.tool_calls_total += 1
        return envelope(False, tool, error="Invalid arguments for %s: %s" % (tool, exc),
                        code="bad_arguments")
    env.tool_calls_total += 1
    env.tool_calls_this_period += 1
    return envelope(True, tool, data=data)

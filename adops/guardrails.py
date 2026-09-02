"""Deterministic guardrails. These run OUTSIDE the model.

Every proposed action passes through `check()` before it can touch the environment.
A rejection returns a machine-readable reason which is fed back to the agent as an
observation, so it can re-plan. Model output can never bypass this layer.
"""
from __future__ import annotations

from typing import Optional, Tuple

from . import metrics as M
from .schema import ACTIVE, PAUSED

ACTION_TOOLS = {"set_budget", "set_status", "launch_creative_test"}
EXPECTED_KEYS = ("entity_id", "metric", "direction", "magnitude_pct", "periods")
ALLOWED_METRICS = {"cpa", "ctr", "cvr", "cpc", "cpm", "conversions", "spend", "roas", "frequency"}


class Rejection(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _clicks_recent(env, entity_id: str, periods: int = 3) -> float:
    first = max(1, env.period - periods + 1)
    return M.sum_raw(env.entity_rows(entity_id, first, env.period))["clicks"]


def check_expected_effect(expected: dict) -> None:
    if not isinstance(expected, dict):
        raise Rejection("expected_effect_missing",
                        "Consequential actions require an expected_effect object.")
    missing = [k for k in EXPECTED_KEYS if expected.get(k) in (None, "")]
    if missing:
        raise Rejection("expected_effect_incomplete",
                        "expected_effect is missing required keys: %s" % ", ".join(missing))
    if expected["metric"] not in ALLOWED_METRICS:
        raise Rejection("expected_effect_invalid_metric",
                        "metric must be one of %s" % sorted(ALLOWED_METRICS))
    if str(expected["direction"]) not in ("increase", "decrease"):
        raise Rejection("expected_effect_invalid_direction",
                        "direction must be 'increase' or 'decrease'")
    try:
        mag = float(expected["magnitude_pct"])
        per = int(expected["periods"])
    except (TypeError, ValueError):
        raise Rejection("expected_effect_invalid_types",
                        "magnitude_pct must be a number and periods an integer")
    if not (1.0 <= mag <= 300.0):
        raise Rejection("expected_effect_implausible",
                        "magnitude_pct must be between 1 and 300")
    if not (1 <= per <= 4):
        raise Rejection("expected_effect_invalid_horizon",
                        "periods must be between 1 and 4")


def check_budgets(env) -> None:
    """Called before every run so that violations cannot be created out of band."""
    pass


def check(env, tool: str, args: dict) -> None:
    """Raise Rejection if the action is not permitted. Otherwise return None."""
    L = env.limits
    acct = env.account

    if env.tool_calls_total >= L.max_total_tool_calls:
        raise Rejection("tool_call_budget_exhausted",
                        "Total tool-call budget (%d) exhausted." % L.max_total_tool_calls)
    if tool in ACTION_TOOLS or tool in ("get_account_snapshot", "get_timeseries",
                                        "compare_segments"):
        if env.tool_calls_this_period >= L.max_tool_calls_per_period:
            raise Rejection("period_tool_call_budget_exhausted",
                            "Tool-call budget for this period (%d) is exhausted; "
                            "call advance_period or finish." % L.max_tool_calls_per_period)

    if tool in ACTION_TOOLS:
        if env.changes_total >= L.max_total_changes:
            raise Rejection("change_budget_exhausted",
                            "Total change budget (%d structural changes) is exhausted."
                            % L.max_total_changes)
        if env.changes_this_period >= L.max_changes_per_period:
            raise Rejection("period_change_budget_exhausted",
                            "Already made %d changes this period (limit %d). Advance the "
                            "period to observe their effect before changing more."
                            % (env.changes_this_period, L.max_changes_per_period))
        check_expected_effect(args.get("expected_effect"))
        if not str(args.get("rationale", "")).strip():
            raise Rejection("rationale_required",
                            "Consequential actions require a non-empty rationale.")

    if tool == "set_budget":
        aid = args.get("entity_id")
        if acct.entity_type(aid) != "ad_set":
            raise Rejection("invalid_entity", "set_budget requires an ad_set id; got %r" % aid)
        ad_set = acct.ad_sets[aid]
        if ad_set.status != ACTIVE:
            raise Rejection("entity_paused", "%s is paused; resume it before changing budget." % aid)
        try:
            new = float(args.get("new_daily_budget"))
        except (TypeError, ValueError):
            raise Rejection("invalid_budget", "new_daily_budget must be a number.")
        old = ad_set.daily_budget
        if new <= 0:
            raise Rejection("invalid_budget", "new_daily_budget must be positive.")
        rel = abs(new - old) / old if old else 1.0
        if rel > L.max_budget_change_pct + 1e-9:
            raise Rejection("budget_step_too_large",
                            "Requested change is %.1f%% of current (%.2f -> %.2f); the limit "
                            "is %.0f%% per action." % (rel * 100, old, new,
                                                       L.max_budget_change_pct * 100))
        if new < L.min_daily_budget:
            raise Rejection("below_min_daily_budget",
                            "Minimum daily budget per ad set is %.2f." % L.min_daily_budget)
        if new > L.max_daily_budget:
            raise Rejection("above_max_daily_budget",
                            "Maximum daily budget per ad set is %.2f." % L.max_daily_budget)
        projected_account = acct.account_daily_budget() - old + new
        if projected_account > L.max_account_daily_budget + 1e-9:
            raise Rejection("account_budget_ceiling",
                            "Account daily budget would become %.2f, above the ceiling %.2f."
                            % (projected_account, L.max_account_daily_budget))
        camp_total = sum(a.daily_budget for a in acct.active_ad_sets(ad_set.campaign_id))
        if camp_total - old + new < L.min_campaign_daily_budget:
            raise Rejection("campaign_floor_breach",
                            "Campaign %s would fall to %.2f/day, below the floor %.2f."
                            % (ad_set.campaign_id, camp_total - old + new,
                               L.min_campaign_daily_budget))
        projected_spend = (env.spend_to_date()
                           + projected_account * 7.0 * env.periods_remaining())
        if projected_spend > env.objective.total_budget + 1e-6:
            raise Rejection("pacing_guard",
                            "At %.2f/day the run would spend about %.0f in total, above the "
                            "budget cap of %.0f. Fund an increase by cutting elsewhere first."
                            % (projected_account, projected_spend, env.objective.total_budget))
        if new < old:
            clicks = _clicks_recent(env, aid)
            if clicks < L.min_clicks_for_budget_cut:
                raise Rejection("insufficient_data_for_cut",
                                "%s has only %d clicks in the last 3 periods; %d are required "
                                "before cutting its budget." % (aid, clicks,
                                                                L.min_clicks_for_budget_cut))

    elif tool == "set_status":
        eid = args.get("entity_id")
        kind = acct.entity_type(eid)
        status = args.get("status")
        if kind not in ("ad_set", "creative"):
            raise Rejection("invalid_entity",
                            "set_status requires an ad_set or creative id; got %r" % eid)
        if status not in (ACTIVE, PAUSED):
            raise Rejection("invalid_status", "status must be 'active' or 'paused'.")
        current = (acct.ad_sets[eid].status if kind == "ad_set"
                   else acct.creatives[eid].status)
        if current == status:
            raise Rejection("no_op", "%s is already %s." % (eid, status))
        if status == PAUSED:
            clicks = _clicks_recent(env, eid)
            if clicks < L.min_clicks_for_pause:
                raise Rejection("insufficient_data_for_pause",
                                "%s has only %d clicks in the last 3 periods; %d are required "
                                "before pausing it." % (eid, clicks, L.min_clicks_for_pause))
            if kind == "ad_set":
                siblings = [a for a in acct.active_ad_sets(acct.ad_sets[eid].campaign_id)
                            if a.id != eid]
                if not siblings:
                    raise Rejection("last_active_ad_set",
                                    "%s is the only active ad set in its campaign." % eid)
                camp_total = sum(a.daily_budget for a in siblings)
                if camp_total < L.min_campaign_daily_budget:
                    raise Rejection("campaign_floor_breach",
                                    "Pausing %s would leave campaign %s at %.2f/day, below the "
                                    "floor %.2f." % (eid, acct.ad_sets[eid].campaign_id,
                                                     camp_total, L.min_campaign_daily_budget))
            else:
                ad_set_id = acct.creatives[eid].ad_set_id
                others = [c for c in acct.active_creatives(ad_set_id) if c.id != eid]
                if not others:
                    raise Rejection("last_active_creative",
                                    "%s is the only active creative in %s." % (eid, ad_set_id))

    elif tool == "launch_creative_test":
        aid = args.get("ad_set_id")
        if acct.entity_type(aid) != "ad_set":
            raise Rejection("invalid_entity",
                            "launch_creative_test requires an ad_set id; got %r" % aid)
        if acct.ad_sets[aid].status != ACTIVE:
            raise Rejection("entity_paused", "%s is paused." % aid)
        if env.creative_test_count >= L.max_active_creative_tests:
            raise Rejection("creative_test_budget_exhausted",
                            "Creative-test budget (%d) exhausted." % L.max_active_creative_tests)
        existing = [c for c in acct.ad_sets[aid].creative_ids
                    if acct.creatives[c].is_test and acct.creatives[c].status == ACTIVE]
        if existing:
            raise Rejection("test_already_running",
                            "%s already has an active creative test (%s); let it gather data."
                            % (aid, existing[0]))

    elif tool in ("advance_period", "finish", "get_account_snapshot", "get_timeseries",
                  "compare_segments"):
        pass
    else:
        raise Rejection("unknown_tool", "No such tool: %r" % tool)

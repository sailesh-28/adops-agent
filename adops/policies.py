"""Decision policies. All three emit the same Decision objects and run through the
same runtime, tool layer and guardrails -- only the reasoning differs."""
from __future__ import annotations

from typing import List, Optional

from . import tools
from .agent import Decision, decide as llm_decide
from .llm import LLMBackend, LLMError
from .schema import ACTIVE, PAUSED


class NoopPolicy:
    """Lower bound: never intervene. Whatever the account does, it does by itself."""
    name = "noop"

    def decide(self, env, briefing, recent, new_rejections, note="") -> Decision:
        return Decision(tool="advance_period", args={"note": "no intervention"},
                        decision="wait", observation="Do-nothing baseline.")


class RulePolicy:
    """A reasonable deterministic auto-optimiser -- the kind of rule set an agency
    would actually run. It checks data sufficiency and statistical significance
    before acting, inspects creatives before blaming an ad set, and reallocates
    budget from losers to winners.

    Its structural limits are the interesting part: it has no memory of whether its
    previous change worked, it cannot sequence a multi-step investigation, and it
    applies fixed thresholds regardless of context.
    """
    name = "rules"

    CUT_PCT = 0.20
    RAISE_PCT = 0.20
    BAD_CPA_MULT = 1.25
    GOOD_CPA_MULT = 0.85
    HIGH_FREQUENCY = 2.5

    COOLDOWN_PERIODS = 2          # let a changed entity finish learning before touching it

    def __init__(self):
        self.last_changed = {}    # entity_id -> live period of the last change
        self.phase = 0
        self.snapshot = None
        self.creative_view = None
        self.suspect = None
        self.cut_made = False

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _last(recent, tool):
        for r in reversed(recent):
            if r["tool"] == tool and "error" not in (r.get("result") or {}):
                return r["result"]
        return None

    def _reset_period(self):
        self.phase = 0
        self.snapshot = None
        self.creative_view = None
        self.suspect = None
        self.cut_made = False

    def _cool(self, entity_id: str, period: int) -> bool:
        """True if this entity may be changed now."""
        last = self.last_changed.get(entity_id)
        return last is None or (period - last) >= self.COOLDOWN_PERIODS

    def _advance(self, why) -> Decision:
        self._reset_period()
        return Decision(tool="advance_period", args={"note": why}, decision="wait",
                        observation=why)

    # -- policy -------------------------------------------------------------
    def decide(self, env, briefing, recent, new_rejections, note="") -> Decision:
        target = briefing["objective"]["target_cpa"]
        budgets = briefing["budgets"]
        if new_rejections and self.phase >= 2:
            return self._advance("previous action was rejected; waiting for new data")

        if self.phase == 0:
            self.phase = 1
            return Decision(tool="get_account_snapshot", args={"lookback_periods": 3},
                            decision="investigate",
                            observation="Routine review of account state.")

        if self.phase == 1:
            self.snapshot = self._last(recent, "get_account_snapshot")
            if not self.snapshot:
                return self._advance("no snapshot available")
            period = briefing["period"]
            bad = [r for r in self.snapshot["ad_sets"]
                   if r["status"] == ACTIVE and r["cpa"]
                   and r["cpa"] > target * self.BAD_CPA_MULT
                   and r["data_sufficiency"] in ("adequate", "thin")
                   and self._cool(r["ad_set_id"], period)]
            bad.sort(key=lambda r: -r["cpa"])
            self.suspect = bad[0] if bad else None
            if self.suspect and self.suspect["active_creatives"] >= 2:
                self.phase = 2
                return Decision(
                    tool="compare_segments",
                    args={"level": "creative", "parent_id": self.suspect["ad_set_id"],
                          "lookback_periods": 3},
                    decision="investigate",
                    observation="%s is above the CPA threshold; checking its creatives."
                                % self.suspect["ad_set_id"])
            self.phase = 3
            return self.decide(env, briefing, recent, [], note)

        if self.phase == 2:
            self.creative_view = self._last(recent, "compare_segments")
            self.phase = 3
            return self.decide(env, briefing, recent, [], note)

        if self.phase == 3:
            self.phase = 4
            if budgets["changes_used_this_period"] >= budgets["changes_max_this_period"]:
                return self._advance("change budget for this period is used")

            # (a) a creative that is significantly worse than its siblings
            if self.creative_view:
                segs = [s for s in self.creative_view["segments"] if s["status"] == ACTIVE]
                if len(segs) >= 2:
                    # Creative-level CPA is too thin to trust: judge creatives on CTR/CPC,
                    # which share the ad set's conversion rate and have far more samples.
                    segs = sorted(segs, key=lambda s: -(s["cpm"] or 0) / max(s["ctr_pct"] or 1e-9, 1e-9))
                    worst, best = segs[0], segs[-1]
                    worst_cpc = (worst["cpm"] or 0) / max(worst["ctr_pct"] or 1e-9, 1e-9) / 10.0
                    best_cpc = (best["cpm"] or 0) / max(best["ctr_pct"] or 1e-9, 1e-9) / 10.0
                    if (worst_cpc > best_cpc * 1.20
                            and worst["ctr_vs_rest"]["verdict"] == "significant"
                            and worst["clicks"] >= env.limits.min_clicks_for_pause
                            and self._cool(worst["entity_id"], briefing["period"])):
                        self.last_changed[worst["entity_id"]] = briefing["period"]
                        self.last_changed[self.suspect["ad_set_id"]] = briefing["period"]
                        return Decision(
                            tool="set_status",
                            args={"entity_id": worst["entity_id"], "status": PAUSED,
                                  "rationale": ("Effective CPC %.2f vs %.2f for the best "
                                                "creative in the ad set, with a "
                                                "significantly lower CTR."
                                                % (worst_cpc, best_cpc)),
                                  "expected_effect": {
                                      "entity_id": self.suspect["ad_set_id"],
                                      "metric": "cpa", "direction": "decrease",
                                      "magnitude_pct": 15, "periods": 2}},
                            decision="act",
                            observation="Pausing the weakest creative in %s."
                                        % self.suspect["ad_set_id"])

            # (b) cut an ad set that is expensive or over-frequency
            cand = self.suspect
            if cand is None and self.snapshot:
                over = [r for r in self.snapshot["ad_sets"]
                        if r["status"] == ACTIVE and (r["frequency"] or 0) > self.HIGH_FREQUENCY
                        and (r["cpm_delta_pct"] or 0) > 10]
                cand = over[0] if over else None
            if cand:
                new_budget = round(cand["daily_budget"] * (1 - self.CUT_PCT), 2)
                if new_budget >= env.limits.min_daily_budget:
                    self.cut_made = True
                    self.last_changed[cand["ad_set_id"]] = briefing["period"]
                    return Decision(
                        tool="set_budget",
                        args={"entity_id": cand["ad_set_id"], "new_daily_budget": new_budget,
                              "rationale": ("CPA %.2f is above the %.0f%% threshold over the "
                                            "target of %.2f." % (cand["cpa"] or 0,
                                                                 self.BAD_CPA_MULT * 100,
                                                                 target)),
                              "expected_effect": {"entity_id": cand["ad_set_id"],
                                                  "metric": "cpa", "direction": "decrease",
                                                  "magnitude_pct": 12, "periods": 2}},
                        decision="act",
                        observation="Cutting budget on %s." % cand["ad_set_id"])
            return self.decide(env, briefing, recent, [], note)

        if self.phase == 4:
            self.phase = 5
            if (budgets["changes_used_this_period"] < budgets["changes_max_this_period"]
                    and self.snapshot):
                good = [r for r in self.snapshot["ad_sets"]
                        if r["status"] == ACTIVE and r["cpa"]
                        and r["cpa"] < target * self.GOOD_CPA_MULT
                        and r["data_sufficiency"] == "adequate"
                        and self._cool(r["ad_set_id"], briefing["period"])]
                good.sort(key=lambda r: r["cpa"])
                if good:
                    win = good[0]
                    new_budget = round(win["daily_budget"] * (1 + self.RAISE_PCT), 2)
                    self.last_changed[win["ad_set_id"]] = briefing["period"]
                    return Decision(
                        tool="set_budget",
                        args={"entity_id": win["ad_set_id"], "new_daily_budget": new_budget,
                              "rationale": ("CPA %.2f is well below the target of %.2f; "
                                            "reallocating budget towards it."
                                            % (win["cpa"], target)),
                              "expected_effect": {"entity_id": win["ad_set_id"],
                                                  "metric": "conversions",
                                                  "direction": "increase",
                                                  "magnitude_pct": 12, "periods": 2}},
                        decision="act",
                        observation="Scaling the best-performing ad set %s."
                                    % win["ad_set_id"])
            return self._advance("no further rule fired this period")

        return self._advance("period complete")


class LLMPolicy:
    """The agent. Chooses its own investigation path and actions."""
    name = "llm"

    def __init__(self, backend: LLMBackend, max_llm_calls: int = 70):
        self.backend = backend
        self.max_llm_calls = max_llm_calls
        self.calls = 0

    def decide(self, env, briefing, recent, new_rejections, note="") -> Decision:
        if self.calls >= self.max_llm_calls:
            return Decision(tool="advance_period",
                            args={"note": "model-call budget exhausted"},
                            decision="wait",
                            observation="Model-call budget exhausted; letting the "
                                        "remaining periods run unchanged.")
        self.calls += 1
        return llm_decide(self.backend, briefing, recent, new_rejections, note)


def build(name: str, backend: Optional[LLMBackend] = None, **kwargs):
    if name == "noop":
        return NoopPolicy()
    if name == "rules":
        return RulePolicy()
    if name == "llm":
        if backend is None:
            raise ValueError("the llm policy requires a backend")
        return LLMPolicy(backend, **kwargs)
    raise ValueError("unknown policy %r" % name)


class OraclePolicy:
    """EVALUATION ONLY -- never used as an agent and never shown scenario labels at
    runtime by any other policy.

    Applies the known-correct intervention for each scenario. Its purpose is to
    validate the environment: if the oracle cannot beat the do-nothing baseline on
    a scenario, that scenario does not reward correct diagnosis and the result for
    every other policy is meaningless. It also provides the regret reference.
    """
    name = "oracle"

    PLANS = {
        # period -> list of (tool, args-without-boilerplate)
        "S1_creative_fatigue": {
            0: [("set_status", {"entity_id": "cr_broad_a", "status": PAUSED}),
                ("launch_creative_test", {"ad_set_id": "as_broad"})],
        },
        "S2_audience_saturation": {
            0: [("cut", {"entity_id": "as_cart", "pct": 0.25}),
                ("raise", {"entity_id": "as_lal", "pct": 0.20})],
            2: [("cut", {"entity_id": "as_cart", "pct": 0.25}),
                ("raise", {"entity_id": "as_broad", "pct": 0.15})],
        },
        "S3_misallocation": {
            0: [("cut", {"entity_id": "as_interior", "pct": 0.25}),
                ("raise", {"entity_id": "as_lal", "pct": 0.25})],
            2: [("cut", {"entity_id": "as_interior", "pct": 0.25}),
                ("raise", {"entity_id": "as_lal", "pct": 0.25})],
            4: [("cut", {"entity_id": "as_interior", "pct": 0.25}),
                ("raise", {"entity_id": "as_lal", "pct": 0.25})],
        },
        "S4_noise_trap": {},
    }

    def __init__(self, scenario_key: str):
        self.plan = self.PLANS.get(scenario_key, {})
        self.queue = []
        self.loaded = -1

    def decide(self, env, briefing, recent, new_rejections, note="") -> Decision:
        period = briefing["period"]
        if self.loaded != period:
            self.loaded = period
            self.queue = list(self.plan.get(period, []))
        if not self.queue:
            return Decision(tool="advance_period", args={"note": "oracle: nothing due"},
                            decision="wait")
        kind, spec = self.queue.pop(0)
        eff = {"entity_id": spec.get("entity_id", spec.get("ad_set_id")),
               "metric": "cpa", "direction": "decrease", "magnitude_pct": 12, "periods": 2}
        if kind in ("cut", "raise"):
            aid = spec["entity_id"]
            cur = env.account.ad_sets[aid].daily_budget
            factor = (1 - spec["pct"]) if kind == "cut" else (1 + spec["pct"])
            if kind == "raise":
                eff = dict(eff, metric="conversions", direction="increase")
            return Decision(tool="set_budget",
                            args={"entity_id": aid,
                                  "new_daily_budget": round(cur * factor, 2),
                                  "rationale": "oracle plan", "expected_effect": eff},
                            decision="act")
        return Decision(tool=kind, args=dict(spec, rationale="oracle plan",
                                             expected_effect=eff), decision="act")

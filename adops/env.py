"""AdAccountEnv: the observable environment wrapped around hidden ground truth.

Owns: account state, period advance, action application, spend accounting,
the action ledger (expected vs realised) and deterministic scoring.
"""
from __future__ import annotations

import datetime as _dt
import zlib
from dataclasses import dataclass, field, replace
from random import Random
from typing import Dict, List, Optional

from . import metrics as M
from . import scenarios as SC
from .schema import ACTIVE, PAUSED, Account, Creative, Limits, Objective
from .world import World, simulate_ad_set

ANCHOR_DATE = _dt.date(2025, 1, 6)      # period 1 starts here (a Monday)
DELAY_FRACTION = 0.25                   # share of conversions reported one period late


@dataclass
class LedgerEntry:
    seq: int
    period: int                     # live period during which the action was taken
    tool: str
    args: dict
    rationale: str
    expected: dict
    baseline: Optional[dict] = None
    realised: Optional[dict] = None
    verdict: str = "pending"
    detail: str = ""

    def summary(self) -> str:
        exp = self.expected or {}
        return ("#%d p%d %s(%s) expect %s %s %s%% within %sp -> %s%s"
                % (self.seq, self.period, self.tool,
                   ",".join("%s=%s" % (k, v) for k, v in sorted(self.args.items())),
                   exp.get("entity_id", "?"), exp.get("metric", "?"),
                   exp.get("direction", "?")[:3] + " " + str(exp.get("magnitude_pct", "?")),
                   exp.get("periods", "?"), self.verdict,
                   (": " + self.detail) if self.detail else ""))


class AdAccountEnv:
    def __init__(self, scenario: SC.Scenario, seed: int = 0, limits: Optional[Limits] = None):
        self.scenario = scenario
        self.seed = seed
        self.limits = limits or Limits()
        self.objective: Objective = scenario.objective
        # NOTE: never use hash() here -- Python randomises string hashing per process,
        # which would silently break cross-process reproducibility.
        self.rng = Random(zlib.crc32(scenario.key.encode("utf-8")) * 1000 + seed)
        self.account: Account = SC.build_account(scenario)
        self.starting_daily_budget = self.account.account_daily_budget()
        if self.limits.max_account_daily_budget is None:
            self.limits = replace(self.limits,
                                  max_account_daily_budget=round(self.starting_daily_budget * 1.15, 2))
        self.world: World = SC.build_world(scenario, self.rng)

        self.rows: List[dict] = []              # reported rows (delayed + noisy)
        self.true_rows: List[dict] = []         # undelayed truth, for evaluation only
        self._carryover: Dict[str, dict] = {}
        self.period: int = 0                    # absolute period index of last completed
        self.live_start_period: int = 0
        self.ledger: List[LedgerEntry] = []
        self.changes_total: int = 0
        self.changes_this_period: int = 0
        self.tool_calls_this_period: int = 0
        self.tool_calls_total: int = 0
        self.rejections: List[dict] = []
        self.creative_test_count: int = 0
        self._seq = 0
        self.finished_reason: Optional[str] = None
        self._burn_in()

    # ------------------------------------------------------------ lifecycle
    def _burn_in(self) -> None:
        for _ in range(self.scenario.burn_in_periods):
            self._simulate_one_period()
        self.live_start_period = self.period

    @property
    def live_period(self) -> int:
        """1-based index of the current live period (0 before any live period)."""
        return self.period - self.live_start_period

    def periods_remaining(self) -> int:
        return max(0, self.objective.horizon_periods - self.live_period)

    def is_done(self) -> bool:
        return self.finished_reason is not None or self.periods_remaining() <= 0

    # ---------------------------------------------------------- simulation
    def _date_for(self, period: int) -> str:
        return (ANCHOR_DATE + _dt.timedelta(weeks=period - 1)).isoformat()

    def _simulate_one_period(self) -> None:
        self.period += 1
        date = self._date_for(self.period)
        new_carry: Dict[str, dict] = {}
        for ad_set in self.account.ad_sets.values():
            if ad_set.status != ACTIVE:
                continue
            active_creatives = [c.id for c in self.account.active_creatives(ad_set.id)]
            results = simulate_ad_set(self.world, ad_set.id, ad_set.daily_budget, active_creatives)
            for cid, res in results.items():
                creative = self.account.creatives[cid]
                campaign = self.account.campaigns[ad_set.campaign_id]

                # deterministic reporting delay: part of this period's conversions
                # land in next period's report, and last period's tail lands now.
                carry = self._carryover.get(cid, {"conversions": 0.0, "revenue": 0.0})
                reported_conv = res["conversions"] * (1.0 - DELAY_FRACTION) + carry["conversions"]
                reported_rev = res["revenue"] * (1.0 - DELAY_FRACTION) + carry["revenue"]
                new_carry[cid] = {
                    "conversions": res["conversions"] * DELAY_FRACTION,
                    "revenue": res["revenue"] * DELAY_FRACTION,
                }

                base = dict(
                    period=self.period, date=date, platform=self.account.platform,
                    campaign_id=campaign.id, campaign_name=campaign.name,
                    ad_set_id=ad_set.id, ad_set_name=ad_set.name,
                    creative_id=cid, creative_name=creative.name,
                    spend=round(res["spend"], 2), impressions=res["impressions"],
                    reach=res["reach"], clicks=res["clicks"],
                )
                self.rows.append(dict(base, conversions=round(reported_conv, 2),
                                      revenue=round(reported_rev, 2)))
                self.true_rows.append(dict(base, conversions=res["conversions"],
                                           revenue=round(res["revenue"], 2)))
        self._carryover = new_carry
        self.world.decay()

    def advance(self) -> dict:
        """Advance the simulation one period and score any pending expected effects."""
        if self.periods_remaining() <= 0:
            return {"ok": False, "error": "horizon_reached"}
        self._simulate_one_period()
        self.changes_this_period = 0
        self.tool_calls_this_period = 0
        self._score_pending_effects()
        return {"ok": True, "period": self.live_period,
                "periods_remaining": self.periods_remaining()}

    # ------------------------------------------------------------- queries
    def rows_in(self, first: int, last: int) -> List[dict]:
        return [r for r in self.rows if first <= r["period"] <= last]

    def entity_rows(self, entity_id: str, first: int, last: int,
                    rows: Optional[List[dict]] = None) -> List[dict]:
        rows = self.rows if rows is None else rows
        kind = self.account.entity_type(entity_id)
        key = {"campaign": "campaign_id", "ad_set": "ad_set_id", "creative": "creative_id"}.get(kind)
        if key is None:
            return []
        return [r for r in rows if r[key] == entity_id and first <= r["period"] <= last]

    def entity_metrics(self, entity_id: str, first: int, last: int,
                       rows: Optional[List[dict]] = None) -> dict:
        return M.aggregate(self.entity_rows(entity_id, first, last, rows=rows))

    def entity_metrics_all(self, first: int, last: int) -> dict:
        return M.aggregate(self.rows_in(first, last))

    def live_rows(self) -> List[dict]:
        return [r for r in self.rows if r["period"] > self.live_start_period]

    def true_live_rows(self) -> List[dict]:
        return [r for r in self.true_rows if r["period"] > self.live_start_period]

    def spend_to_date(self) -> float:
        return sum(r["spend"] for r in self.live_rows())

    # ------------------------------------------------------------- actions
    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def _register(self, tool: str, args: dict, rationale: str, expected: dict,
                  entity_id: str) -> LedgerEntry:
        baseline = self._effect_baseline(expected.get("entity_id") or entity_id,
                                         expected.get("metric", "cpa"))
        entry = LedgerEntry(seq=self._next_seq(), period=self.live_period, tool=tool,
                            args=args, rationale=rationale, expected=expected,
                            baseline=baseline)
        self.ledger.append(entry)
        self.changes_total += 1
        self.changes_this_period += 1
        return entry

    def apply_set_budget(self, ad_set_id: str, new_daily_budget: float,
                         rationale: str, expected: dict) -> dict:
        ad_set = self.account.ad_sets[ad_set_id]
        old = ad_set.daily_budget
        ad_set.daily_budget = round(float(new_daily_budget), 2)
        rel = abs(ad_set.daily_budget - old) / old if old else 1.0
        learning = rel >= self.limits.budget_change_learning_threshold
        if learning:
            self.world.start_learning(ad_set_id, self.limits.learning_penalty_periods,
                                      self.limits.learning_efficiency)
        self._register("set_budget",
                       {"entity_id": ad_set_id, "new_daily_budget": ad_set.daily_budget},
                       rationale, expected, ad_set_id)
        return {"ok": True, "entity_id": ad_set_id, "old_daily_budget": old,
                "new_daily_budget": ad_set.daily_budget,
                "relative_change": round(rel, 4),
                "learning_phase_triggered": learning,
                "account_daily_budget": round(self.account.account_daily_budget(), 2)}

    def apply_set_status(self, entity_id: str, status: str, rationale: str,
                         expected: dict) -> dict:
        kind = self.account.entity_type(entity_id)
        if kind == "ad_set":
            self.account.ad_sets[entity_id].status = status
            affected = entity_id
        else:
            self.account.creatives[entity_id].status = status
            affected = self.account.creatives[entity_id].ad_set_id
        self.world.start_learning(affected, self.limits.learning_penalty_periods,
                                  self.limits.learning_efficiency)
        self._register("set_status", {"entity_id": entity_id, "status": status},
                       rationale, expected, entity_id)
        return {"ok": True, "entity_id": entity_id, "entity_type": kind, "status": status,
                "learning_phase_triggered": True, "affected_ad_set": affected}

    def apply_launch_creative_test(self, ad_set_id: str, rationale: str,
                                   expected: dict) -> dict:
        self.creative_test_count += 1
        cid = "cr_test_%s_%d" % (ad_set_id.replace("as_", ""), self.creative_test_count)
        name = "Test Creative %d (%s)" % (self.creative_test_count,
                                          self.account.ad_sets[ad_set_id].name)
        self.account.creatives[cid] = Creative(id=cid, name=name, ad_set_id=ad_set_id,
                                               launched_period=self.live_period, is_test=True)
        self.account.ad_sets[ad_set_id].creative_ids.append(cid)
        self.world.add_creative(cid, ad_set_id)
        self.world.start_learning(ad_set_id, self.limits.learning_penalty_periods,
                                  self.limits.learning_efficiency)
        self._register("launch_creative_test", {"ad_set_id": ad_set_id},
                       rationale, expected, ad_set_id)
        return {"ok": True, "ad_set_id": ad_set_id, "new_creative_id": cid,
                "new_creative_name": name, "learning_phase_triggered": True,
                "note": "New creative quality is unknown until it accumulates data."}

    # ------------------------------------------- expected vs realised effect
    def _metric_value(self, entity_id: str, metric: str, first: int, last: int) -> Optional[float]:
        agg = self.entity_metrics(entity_id, first, last)
        if metric in ("conversions", "spend", "clicks", "impressions", "revenue"):
            return agg.get(metric)
        return agg.get(metric)

    def _effect_baseline(self, entity_id: str, metric: str) -> dict:
        first, last = max(1, self.period - 2), self.period
        return {"entity_id": entity_id, "metric": metric,
                "value": self._metric_value(entity_id, metric, first, last),
                "window": [first, last]}

    def _score_pending_effects(self) -> None:
        for e in self.ledger:
            if e.verdict != "pending":
                continue
            horizon = int(e.expected.get("periods", 2) or 2)
            action_abs_period = self.live_start_period + e.period
            if self.period < action_abs_period + 1:
                continue
            elapsed = self.period - action_abs_period
            if elapsed < horizon and self.periods_remaining() > 0:
                continue  # not yet due
            entity = e.expected.get("entity_id") or e.args.get("entity_id") or e.args.get("ad_set_id")
            metric = e.expected.get("metric", "cpa")
            after = self._metric_value(entity, metric,
                                       action_abs_period + 1,
                                       min(self.period, action_abs_period + horizon))
            base = (e.baseline or {}).get("value")
            e.realised = {"entity_id": entity, "metric": metric, "value": after,
                          "window": [action_abs_period + 1,
                                     min(self.period, action_abs_period + horizon)]}
            if after is None or base in (None, 0):
                e.verdict, e.detail = "unmeasurable", "no comparable data for this metric"
                continue
            change = (after - base) / base
            e.realised["change_pct"] = round(change * 100.0, 1)
            want_up = str(e.expected.get("direction", "")).startswith("incr")
            target = float(e.expected.get("magnitude_pct", 0) or 0) / 100.0
            moved_right_way = (change > 0) if want_up else (change < 0)
            size = abs(change)
            if not moved_right_way:
                e.verdict = "contradicted"
            elif target and size >= 0.5 * target:
                e.verdict = "confirmed"
            else:
                e.verdict = "partial"
            e.detail = ("%s %s: %.4g -> %.4g (%+.1f%%), expected %s %.0f%%"
                        % (entity, metric, base, after, change * 100.0,
                           e.expected.get("direction", "?"),
                           float(e.expected.get("magnitude_pct", 0) or 0)))

    def force_score_effects(self) -> None:
        """At end of run, resolve everything still pending with whatever data exists."""
        for e in self.ledger:
            if e.verdict == "pending":
                e.expected = dict(e.expected, periods=1)
        self._score_pending_effects()

    # -------------------------------------------------------------- scoring
    def score(self) -> dict:
        rows = self.live_rows()
        agg = M.aggregate(rows)
        true_agg = M.aggregate(self.true_live_rows())
        spend = agg["spend"] or 0.0
        conv = true_agg["conversions"] or 0.0        # score on truth, not on delayed report
        cpa = M._div(spend, conv)
        obj = self.objective
        violations = []
        if spend > obj.total_budget * 1.0001:
            violations.append("total_budget_exceeded")
        if cpa is not None and cpa > obj.target_cpa:
            violations.append("cpa_target_missed")
        for camp in self.account.campaigns.values():
            if not self.account.active_ad_sets(camp.id):
                violations.append("campaign_fully_paused:%s" % camp.id)
        cpa_factor = 1.0 if (cpa is None or cpa <= obj.target_cpa) else obj.target_cpa / cpa
        budget_factor = 1.0 if spend <= obj.total_budget else obj.total_budget / spend
        verdicts = {}
        for e in self.ledger:
            verdicts[e.verdict] = verdicts.get(e.verdict, 0) + 1
        return {
            "scenario": self.scenario.key,
            "seed": self.seed,
            "periods_run": self.live_period,
            "conversions": round(conv, 2),
            "spend": round(spend, 2),
            "revenue": round(true_agg["revenue"] or 0.0, 2),
            "cpa": round(cpa, 2) if cpa else None,
            "roas": round(M._div(true_agg["revenue"] or 0.0, spend) or 0.0, 3),
            "target_cpa": obj.target_cpa,
            "total_budget": obj.total_budget,
            "budget_utilisation": round(spend / obj.total_budget, 4) if obj.total_budget else None,
            "objective_score": round(conv * cpa_factor * budget_factor, 2),
            "hard_violations": violations,
            "violation_count": len(violations),
            "interventions": self.changes_total,
            "rejections": len(self.rejections),
            "tool_calls": self.tool_calls_total,
            "effect_verdicts": verdicts,
            "finished_reason": self.finished_reason,
        }

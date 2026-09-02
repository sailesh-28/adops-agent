from __future__ import annotations

import json
import sys
import unittest

sys.path.insert(0, "/Users/sailesh/performance-marketing-agent")

from adops import brief, metrics as M, scenarios as SC, stats as S, tools
from adops.env import AdAccountEnv
from adops.schema import ACTIVE, PAUSED


def make(key="S1_creative_fatigue", seed=7):
    return AdAccountEnv(SC.get(key), seed=seed)


class TestMetrics(unittest.TestCase):
    def test_derive(self):
        d = M.derive({"spend": 100, "impressions": 10000, "reach": 5000,
                      "clicks": 200, "conversions": 10, "revenue": 900})
        self.assertAlmostEqual(d["ctr"], 0.02)
        self.assertAlmostEqual(d["cpc"], 0.5)
        self.assertAlmostEqual(d["cpm"], 10.0)
        self.assertAlmostEqual(d["cvr"], 0.05)
        self.assertAlmostEqual(d["cpa"], 10.0)
        self.assertAlmostEqual(d["roas"], 9.0)
        self.assertAlmostEqual(d["frequency"], 2.0)

    def test_safe_division_returns_none_not_zero(self):
        d = M.derive({"spend": 50, "impressions": 0, "clicks": 0, "conversions": 0,
                      "reach": 0, "revenue": 0})
        for k in ("ctr", "cpc", "cpm", "cvr", "cpa", "frequency"):
            self.assertIsNone(d[k], k)

    def test_aggregate_pools_raw_counts_not_ratios(self):
        rows = [{"spend": 100, "impressions": 1000, "clicks": 10, "conversions": 1,
                 "reach": 900, "revenue": 100},
                {"spend": 900, "impressions": 90000, "clicks": 900, "conversions": 30,
                 "reach": 80000, "revenue": 3000}]
        agg = M.aggregate(rows)
        # naive mean of the two CPAs would be 65; the correct pooled CPA is 1000/31
        self.assertAlmostEqual(agg["cpa"], 1000.0 / 31.0)


class TestStats(unittest.TestCase):
    def test_small_samples_are_refused(self):
        self.assertEqual(S.two_proportion_test(1, 20, 2, 25)["verdict"], "insufficient_data")

    def test_clear_difference_is_significant(self):
        r = S.two_proportion_test(60, 1000, 20, 1000)
        self.assertEqual(r["verdict"], "significant")

    def test_similar_rates_not_significant(self):
        r = S.two_proportion_test(30, 1000, 28, 1000)
        self.assertEqual(r["verdict"], "not_significant")

    def test_trend_sign(self):
        self.assertLess(S.linear_trend([5, 4, 3, 2, 1])["slope"], 0)
        self.assertGreater(S.linear_trend([1, 2, 3, 4, 5])["slope"], 0)


class TestDeterminism(unittest.TestCase):
    def test_same_seed_identical(self):
        a, b = make(seed=3), make(seed=3)
        for _ in range(3):
            a.advance(); b.advance()
        self.assertEqual(a.rows, b.rows)
        self.assertEqual(a.score(), b.score())

    def test_different_seed_differs(self):
        a, b = make(seed=3), make(seed=4)
        for _ in range(3):
            a.advance(); b.advance()
        self.assertNotEqual(a.score()["conversions"], b.score()["conversions"])


class TestEnvironment(unittest.TestCase):
    def test_burn_in_history_exists(self):
        e = make()
        self.assertEqual(e.period, e.scenario.burn_in_periods)
        self.assertEqual(e.live_period, 0)
        self.assertGreater(len(e.rows), 100)

    def test_reporting_delay_defers_conversions(self):
        e = make()
        reported = sum(r["conversions"] for r in e.rows)
        true = sum(r["conversions"] for r in e.true_rows)
        self.assertLess(reported, true)                 # a tail is always outstanding
        self.assertGreater(reported, true * 0.9)

    def test_pausing_ad_set_removes_it_from_delivery(self):
        e = make()
        e.apply_set_status("as_lal", PAUSED, "test", {})
        before = e.period
        e.advance()
        rows = [r for r in e.rows if r["period"] > before and r["ad_set_id"] == "as_lal"]
        self.assertEqual(rows, [])

    def test_budget_increase_raises_spend(self):
        lo, hi = make(seed=11), make(seed=11)
        hi.account.ad_sets["as_lal"].daily_budget *= 1.25
        lo.advance(); hi.advance()
        lo_spend = M.sum_raw(lo.entity_rows("as_lal", lo.period, lo.period))["spend"]
        hi_spend = M.sum_raw(hi.entity_rows("as_lal", hi.period, hi.period))["spend"]
        self.assertGreater(hi_spend, lo_spend * 1.1)

    def test_large_budget_change_triggers_learning_phase(self):
        e = make()
        r = e.apply_set_budget("as_lal", e.account.ad_sets["as_lal"].daily_budget * 1.24,
                               "test", {})
        self.assertTrue(r["learning_phase_triggered"])
        self.assertGreater(e.world.ad_sets["as_lal"].learning_periods, 0)

    def test_small_budget_change_does_not_trigger_learning(self):
        e = make()
        r = e.apply_set_budget("as_lal", e.account.ad_sets["as_lal"].daily_budget * 1.05,
                               "test", {})
        self.assertFalse(r["learning_phase_triggered"])

    def test_learning_phase_costs_conversions(self):
        base, hit = make(seed=21), make(seed=21)
        hit.world.start_learning("as_lal", 1, 0.85)
        base.advance(); hit.advance()
        b = M.sum_raw(base.entity_rows("as_lal", base.period, base.period))["conversions"]
        h = M.sum_raw(hit.entity_rows("as_lal", hit.period, hit.period))["conversions"]
        self.assertLess(h, b)

    def test_creative_test_adds_creative_and_it_delivers(self):
        e = make()
        r = e.apply_launch_creative_test("as_lal", "test", {})
        cid = r["new_creative_id"]
        e.advance()
        rows = [x for x in e.rows if x["period"] == e.period and x["creative_id"] == cid]
        self.assertEqual(len(rows), 1)
        self.assertGreater(rows[0]["impressions"], 0)

    def test_horizon_is_enforced(self):
        e = make()
        for _ in range(e.objective.horizon_periods):
            self.assertTrue(e.advance()["ok"])
        self.assertFalse(e.advance()["ok"])
        self.assertTrue(e.is_done())


class TestGuardrails(unittest.TestCase):
    def setUp(self):
        self.e = make()
        self.eff = {"entity_id": "as_lal", "metric": "cpa", "direction": "decrease",
                    "magnitude_pct": 10, "periods": 2}

    def rej(self, tool, args):
        r = tools.dispatch(self.e, tool, args)
        self.assertFalse(r["ok"], "expected rejection, got %s" % r)
        return r["error_code"]

    def test_budget_step_cap(self):
        self.assertEqual(self.rej("set_budget", {
            "entity_id": "as_lal", "new_daily_budget": 400, "rationale": "x",
            "expected_effect": self.eff}), "budget_step_too_large")

    def test_expected_effect_required(self):
        self.assertEqual(self.rej("set_budget", {
            "entity_id": "as_lal", "new_daily_budget": 100, "rationale": "x"}),
            "expected_effect_missing")

    def test_rationale_required(self):
        self.assertEqual(self.rej("set_budget", {
            "entity_id": "as_lal", "new_daily_budget": 100, "rationale": "  ",
            "expected_effect": self.eff}), "rationale_required")

    def test_unknown_entity(self):
        self.assertEqual(self.rej("set_budget", {
            "entity_id": "as_nope", "new_daily_budget": 100, "rationale": "x",
            "expected_effect": self.eff}), "invalid_entity")

    def test_cannot_pause_last_creative(self):
        e = self.e
        eff = dict(self.eff, entity_id="cr_lal_a")
        tools.dispatch(e, "set_status", {"entity_id": "cr_lal_a", "status": PAUSED,
                                         "rationale": "x", "expected_effect": eff})
        r = tools.dispatch(e, "set_status", {"entity_id": "cr_lal_b", "status": PAUSED,
                                             "rationale": "x", "expected_effect": eff})
        self.assertEqual(r["error_code"], "last_active_creative")

    def test_pause_requires_data(self):
        e = self.e
        e.apply_launch_creative_test("as_lal", "x", self.eff)
        cid = e.account.ad_sets["as_lal"].creative_ids[-1]
        r = tools.dispatch(e, "set_status", {"entity_id": cid, "status": PAUSED,
                                             "rationale": "x",
                                             "expected_effect": dict(self.eff, entity_id=cid)})
        self.assertEqual(r["error_code"], "insufficient_data_for_pause")

    def test_change_budget_per_period(self):
        e = self.e
        for i in range(e.limits.max_changes_per_period):
            r = tools.dispatch(e, "set_budget", {
                "entity_id": "as_lal", "new_daily_budget":
                    round(e.account.ad_sets["as_lal"].daily_budget * 0.9, 2),
                "rationale": "x", "expected_effect": self.eff})
            self.assertTrue(r["ok"], r)
        r = tools.dispatch(e, "set_budget", {
            "entity_id": "as_lal", "new_daily_budget":
                round(e.account.ad_sets["as_lal"].daily_budget * 0.95, 2),
            "rationale": "x", "expected_effect": self.eff})
        self.assertEqual(r["error_code"], "period_change_budget_exhausted")

    def test_pacing_guard_blocks_unfunded_increase(self):
        e = self.e
        # raising a big ad set without cutting anything would blow the budget cap
        r = tools.dispatch(e, "set_budget", {
            "entity_id": "as_broad", "new_daily_budget":
                round(e.account.ad_sets["as_broad"].daily_budget * 1.2, 2),
            "rationale": "x", "expected_effect": dict(self.eff, entity_id="as_broad")})
        self.assertEqual(r["error_code"], "pacing_guard")

    def test_total_change_budget(self):
        e = self.e
        made = 0
        while made < e.limits.max_total_changes:
            r = tools.dispatch(e, "set_budget", {
                "entity_id": "as_lal", "new_daily_budget":
                    round(e.account.ad_sets["as_lal"].daily_budget * 0.95, 2),
                "rationale": "x", "expected_effect": self.eff})
            if r["ok"]:
                made += 1
            else:
                self.assertIn(r["error_code"], ("period_change_budget_exhausted",
                                                "below_min_daily_budget"))
                if r["error_code"] == "below_min_daily_budget":
                    return
                e.advance()
        r = tools.dispatch(e, "set_budget", {
            "entity_id": "as_lal", "new_daily_budget":
                round(e.account.ad_sets["as_lal"].daily_budget * 0.95, 2),
            "rationale": "x", "expected_effect": self.eff})
        self.assertEqual(r["error_code"], "change_budget_exhausted")


class TestNoHiddenStateLeak(unittest.TestCase):
    FORBIDDEN = ("base_cvr", "base_ctr", "base_cpm", "audience_size", "fatigue_rate",
                 "saturation_strength", "efficient_weekly_spend", "learning_periods",
                 "cum_impressions", "correct_action_family")

    def test_tool_outputs_contain_no_ground_truth(self):
        e = make()
        payloads = [
            tools.dispatch(e, "get_account_snapshot", {}),
            tools.dispatch(e, "get_timeseries", {"entity_id": "as_broad"}),
            tools.dispatch(e, "compare_segments", {"level": "ad_set"}),
            tools.dispatch(e, "compare_segments", {"level": "creative",
                                                   "parent_id": "as_broad"}),
            brief.build(e),
        ]
        blob = json.dumps(payloads)
        for key in self.FORBIDDEN:
            self.assertNotIn(key, blob, "hidden field %r leaked into observations" % key)


class TestScenarioSignatures(unittest.TestCase):
    """Each scenario must actually present the signature it claims to."""

    def worst(self, e):
        return brief.ad_set_table(e)[0]["ad_set_id"]

    def test_s1_defective_ad_set_underperforms_the_account(self):
        """as_broad must carry a materially worse CPA than the account average.
        It is deliberately NOT always the single worst ad set: small ad sets often
        look worse by chance, which is what makes the diagnosis non-trivial."""
        ratios = []
        for s in range(1, 9):
            e = make("S1_creative_fatigue", s)
            acct = brief.account_summary(e)["cpa"]
            broad = dict((r["ad_set_id"], r["cpa"]) for r in brief.ad_set_table(e))["as_broad"]
            ratios.append(broad / acct)
        self.assertGreaterEqual(sum(r > 1.15 for r in ratios), 7)
        self.assertGreater(sum(ratios) / len(ratios), 1.25)

    def test_s1_fatigued_creative_underperforms_its_sibling(self):
        wins = 0
        for s in range(1, 9):
            e = make("S1_creative_fatigue", s)
            a = e.entity_metrics("cr_broad_a", e.period - 2, e.period)
            b = e.entity_metrics("cr_broad_b", e.period - 2, e.period)
            wins += a["ctr"] < b["ctr"]
        self.assertGreaterEqual(wins, 7)

    def test_s2_saturated_ad_set_shows_high_frequency_and_cpm(self):
        for s in range(1, 5):
            e = make("S2_audience_saturation", s)
            cart = e.entity_metrics("as_cart", e.period - 2, e.period)
            rt30 = e.entity_metrics("as_rt30", e.period - 2, e.period)
            self.assertGreater(cart["frequency"], rt30["frequency"])
            self.assertGreater(cart["cpm"], rt30["cpm"])

    def test_s3_underfunded_ad_set_converts_better(self):
        wins = 0
        for s in range(1, 9):
            e = make("S3_misallocation", s)
            lal = e.entity_metrics("as_lal", e.period - 2, e.period)
            inte = e.entity_metrics("as_interior", e.period - 2, e.period)
            wins += lal["cvr"] > inte["cvr"]
        self.assertGreaterEqual(wins, 7)

    def test_s4_has_no_latent_quality_gap(self):
        e = make("S4_noise_trap", 1)
        cvrs = {a: e.world.ad_sets[a].base_cvr
                for a in ("as_broad", "as_lal", "as_interior", "as_seas_broad")}
        self.assertEqual(len(set(cvrs.values())), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)

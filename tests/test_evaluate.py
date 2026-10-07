from __future__ import annotations

import json
import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adops import evaluate as E
from adops import policies, runtime, scenarios as SC
from adops.env import AdAccountEnv
from adops.schema import PAUSED


def env(key, seed=2):
    return AdAccountEnv(SC.get(key), seed=seed)


class FakeAdSetBudgetCutPolicy:
    """Touches the correct ad set (as_broad) with a plain budget cut, never the
    creative. Stand-in for the exact failure the S1 seed-1 post-mortem found."""
    name = "fake"

    def __init__(self, entity_id="as_broad"):
        self.entity_id = entity_id
        self.done = False

    def decide(self, env, briefing, recent, new_rejections, note=""):
        if not self.done:
            self.done = True
            return policies.Decision(
                tool="set_budget",
                args={"entity_id": self.entity_id, "new_daily_budget":
                      round(env.account.ad_sets[self.entity_id].daily_budget * 0.85, 2),
                      "rationale": "looks like a drag",
                      "expected_effect": {"entity_id": self.entity_id, "metric": "cpa",
                                          "direction": "decrease", "magnitude_pct": 12,
                                          "periods": 2},
                      "considered_alternative": {"level": "creative",
                                                 "summary": "pause a creative instead",
                                                 "why_rejected": "did not check creative "
                                                                 "significance"},
                      "pattern_check": {"similar_past_action": "none",
                                        "reasoning": "first action"}},
                decision="act")
        return policies.Decision(tool="advance_period", args={"note": "n"}, decision="wait")


class TestSpecParsing(unittest.TestCase):
    def test_simple_spec(self):
        self.assertEqual(E._parse_spec("cut_budget:as_cart"),
                         {"kind": "cut_budget", "entity": "as_cart"})

    def test_shift_spec(self):
        self.assertEqual(E._parse_spec("shift_budget:as_interior->as_lal"),
                         {"kind": "shift_budget", "from": "as_interior", "to": "as_lal"})

    def test_none_spec(self):
        self.assertEqual(E._parse_spec("none"), {"kind": "none"})

    def test_every_scenario_parses_without_error(self):
        for key in SC.keys():
            for spec in SC.get(key).correct_action_family:
                parsed = E._parse_spec(spec)
                self.assertIn("kind", parsed)


class TestMechanismDiagnosis(unittest.TestCase):
    """Post-mortem fix #4: distinguish touching the correct entity by accident from
    identifying the actual failure mechanism."""

    def test_plain_ad_set_budget_cut_on_as_broad_does_not_count_as_diagnosis(self):
        """This is the exact case the post-mortem flagged: the old metric counted
        this as a hit because it only checked entity membership."""
        e = env("S1_creative_fatigue")
        log = runtime.run_episode(e, FakeAdSetBudgetCutPolicy("as_broad"))
        mech = E.mechanism_diagnosis(SC.get("S1_creative_fatigue"), e, log)
        self.assertFalse(mech["acted_on_mechanism"])
        self.assertFalse(mech["diagnosis_hit"])

    def test_pausing_the_actual_fatigued_creative_counts_as_diagnosis(self):
        e = env("S1_creative_fatigue")

        class PauseCreative:
            name = "fake"
            def __init__(self):
                self.done = False
            def decide(self, env, briefing, recent, new_rejections, note=""):
                if not self.done:
                    self.done = True
                    return policies.Decision(
                        tool="set_status",
                        args={"entity_id": "cr_broad_a", "status": "paused",
                              "rationale": "r",
                              "expected_effect": {"entity_id": "as_broad", "metric": "cpa",
                                                  "direction": "decrease",
                                                  "magnitude_pct": 10, "periods": 2},
                              "considered_alternative": {"level": "ad_set", "summary": "s",
                                                         "why_rejected": "r"},
                              "pattern_check": {"similar_past_action": "none",
                                                "reasoning": "first"}},
                        decision="act")
                return policies.Decision(tool="advance_period", args={"note": "n"},
                                         decision="wait")

        log = runtime.run_episode(e, PauseCreative())
        mech = E.mechanism_diagnosis(SC.get("S1_creative_fatigue"), e, log)
        self.assertTrue(mech["acted_on_mechanism"])

    def test_oracle_acts_on_mechanism_for_every_defective_scenario(self):
        for key in ("S1_creative_fatigue", "S2_audience_saturation", "S3_misallocation"):
            e = env(key)
            log = runtime.run_episode(e, policies.OraclePolicy(key))
            mech = E.mechanism_diagnosis(SC.get(key), e, log)
            self.assertTrue(mech["acted_on_mechanism"], key)

    def test_noise_trap_mechanism_hit_means_restraint(self):
        e = env("S4_noise_trap")
        log = runtime.run_episode(e, policies.NoopPolicy())
        mech = E.mechanism_diagnosis(SC.get("S4_noise_trap"), e, log)
        self.assertTrue(mech["diagnosis_hit"])

        e2 = env("S4_noise_trap")
        log2 = runtime.run_episode(e2, policies.RulePolicy())
        mech2 = E.mechanism_diagnosis(SC.get("S4_noise_trap"), e2, log2)
        self.assertFalse(mech2["acted_on_mechanism"])   # it does intervene
        self.assertFalse(mech2["diagnosis_hit"])

    def test_wrong_direction_budget_change_does_not_satisfy_cut_spec(self):
        """A RAISE on as_cart must not satisfy "cut_budget:as_cart"."""
        e = env("S2_audience_saturation")

        class RaiseCart:
            name = "fake"
            def __init__(self):
                self.done = False
            def decide(self, env, briefing, recent, new_rejections, note=""):
                if not self.done:
                    self.done = True
                    return policies.Decision(
                        tool="set_budget",
                        args={"entity_id": "as_cart", "new_daily_budget":
                              round(env.account.ad_sets["as_cart"].daily_budget * 1.15, 2),
                              "rationale": "r",
                              "expected_effect": {"entity_id": "as_cart", "metric": "cpa",
                                                  "direction": "decrease",
                                                  "magnitude_pct": 10, "periods": 2},
                              "considered_alternative": {"level": "creative", "summary": "s",
                                                         "why_rejected": "r"},
                              "pattern_check": {"similar_past_action": "none",
                                                "reasoning": "first"}},
                        decision="act")
                return policies.Decision(tool="advance_period", args={"note": "n"},
                                         decision="wait")

        log = runtime.run_episode(e, RaiseCart())
        mech = E.mechanism_diagnosis(SC.get("S2_audience_saturation"), e, log)
        self.assertFalse(mech["acted_on_mechanism"])

    def test_shift_budget_requires_both_the_cut_and_the_raise(self):
        e = env("S3_misallocation")

        class OnlyCut:
            name = "fake"
            def __init__(self):
                self.done = False
            def decide(self, env, briefing, recent, new_rejections, note=""):
                if not self.done:
                    self.done = True
                    return policies.Decision(
                        tool="set_budget",
                        args={"entity_id": "as_interior", "new_daily_budget":
                              round(env.account.ad_sets["as_interior"].daily_budget * 0.8, 2),
                              "rationale": "r",
                              "expected_effect": {"entity_id": "as_interior", "metric": "cpa",
                                                  "direction": "decrease",
                                                  "magnitude_pct": 10, "periods": 2},
                              "considered_alternative": {"level": "creative", "summary": "s",
                                                         "why_rejected": "r"},
                              "pattern_check": {"similar_past_action": "none",
                                                "reasoning": "first"}},
                        decision="act")
                return policies.Decision(tool="advance_period", args={"note": "n"},
                                         decision="wait")

        log = runtime.run_episode(e, OnlyCut())
        mech = E.mechanism_diagnosis(SC.get("S3_misallocation"), e, log)
        self.assertFalse(mech["acted_on_mechanism"],
                         "a shift requires BOTH the cut and the raise half")

    def test_investigation_at_wrong_level_does_not_count(self):
        """get_timeseries on the ad set (not the creative) must not satisfy a
        creative-level investigation requirement, even though the ad set is one of
        the scenario's named entities (via the creative_test:as_broad spec)."""
        e = env("S1_creative_fatigue")
        steps = [SimpleNamespace(tool="get_timeseries", args={"entity_id": "as_broad"},
                                 ok=True)]
        log = SimpleNamespace(steps=steps, ledger=[])
        self.assertFalse(E._investigated_at_level(log, "creative", {"as_broad", "cr_broad_a"},
                                                   e.account))

    def test_investigation_at_correct_level_counts(self):
        e = env("S1_creative_fatigue")
        steps = [SimpleNamespace(tool="get_timeseries", args={"entity_id": "cr_broad_a"},
                                 ok=True)]
        log = SimpleNamespace(steps=steps, ledger=[])
        self.assertTrue(E._investigated_at_level(log, "creative", {"as_broad", "cr_broad_a"},
                                                  e.account))

    def test_compare_segments_at_creative_level_with_matching_parent_counts(self):
        e = env("S1_creative_fatigue")
        steps = [SimpleNamespace(tool="compare_segments",
                                 args={"level": "creative", "parent_id": "as_broad"},
                                 ok=True)]
        log = SimpleNamespace(steps=steps, ledger=[])
        self.assertTrue(E._investigated_at_level(log, "creative", {"as_broad", "cr_broad_a"},
                                                  e.account))

    def test_rejected_tool_calls_do_not_count_as_investigation(self):
        e = env("S1_creative_fatigue")
        steps = [SimpleNamespace(tool="get_timeseries", args={"entity_id": "cr_broad_a"},
                                 ok=False)]
        log = SimpleNamespace(steps=steps, ledger=[])
        self.assertFalse(E._investigated_at_level(log, "creative", {"as_broad", "cr_broad_a"},
                                                   e.account))


class TestReplaysPriorSavedTrace(unittest.TestCase):
    """Regression check for the original S1 seed-1 post-mortem finding: a trace that
    only ever investigates and acts at the ad-set level must score as a miss.
    Self-contained fixture (the steps below mirror that trace's tool calls) so the
    test does not depend on a mutable, gitignored runs/ file."""

    STEPS = [
        ("compare_segments", {"level": "ad_set", "lookback_periods": 3}),
        ("get_timeseries", {"entity_id": "as_broad", "periods": 8}),
        ("set_budget", {"entity_id": "as_broad", "new_daily_budget": 125.0}),
        ("set_budget", {"entity_id": "as_rt30", "new_daily_budget": 59.0}),
        ("compare_segments", {"level": "ad_set", "parent_id": "cmp_seasonal"}),
        ("set_budget", {"entity_id": "as_seas_broad", "new_daily_budget": 35.0}),
        ("set_status", {"entity_id": "as_seas_broad", "status": "paused"}),
        ("set_budget", {"entity_id": "as_lal", "new_daily_budget": 113.0}),
    ]

    def test_ad_set_only_trace_is_scored_as_a_miss(self):
        steps = [SimpleNamespace(tool=t, args=a, ok=True) for t, a in self.STEPS]
        log = SimpleNamespace(steps=steps, ledger=[])
        scenario = SC.get("S1_creative_fatigue")
        e = env("S1_creative_fatigue", seed=1)
        mech = E.mechanism_diagnosis(scenario, e, log)
        self.assertFalse(mech["diagnosis_hit"])
        self.assertFalse(mech["acted_on_mechanism"],
                         "touching as_broad with a budget cut is not the mechanism: the "
                         "correct fix is pausing cr_broad_a or testing a new creative")


class TestReproducibility(unittest.TestCase):
    def test_same_seed_same_mechanism_score_across_processes(self):
        import subprocess
        script = (
            "import sys; sys.path.insert(0, '.'); "
            "from adops import evaluate as E, policies, runtime, scenarios as SC; "
            "from adops.env import AdAccountEnv; "
            "e = AdAccountEnv(SC.get('S2_audience_saturation'), seed=4); "
            "log = runtime.run_episode(e, policies.RulePolicy()); "
            "mech = E.mechanism_diagnosis(SC.get('S2_audience_saturation'), e, log); "
            "print(log.score['objective_score'], mech['diagnosis_hit'])"
        )
        out1 = subprocess.run([sys.executable, "-c", script], capture_output=True,
                              text=True, cwd=".")
        out2 = subprocess.run([sys.executable, "-c", script], capture_output=True,
                              text=True, cwd=".")
        self.assertEqual(out1.returncode, 0, out1.stderr)
        self.assertEqual(out1.stdout, out2.stdout)

    def test_full_episode_deterministic_given_seed(self):
        a = env("S3_misallocation", seed=9)
        b = env("S3_misallocation", seed=9)
        log_a = runtime.run_episode(a, policies.RulePolicy())
        log_b = runtime.run_episode(b, policies.RulePolicy())
        self.assertEqual(log_a.score, log_b.score)
        self.assertEqual([e["verdict"] for e in log_a.ledger],
                         [e["verdict"] for e in log_b.ledger])
        self.assertEqual([e["considered_alternative"] for e in log_a.ledger],
                         [e["considered_alternative"] for e in log_b.ledger])


if __name__ == "__main__":
    unittest.main(verbosity=2)

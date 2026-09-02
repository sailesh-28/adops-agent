from __future__ import annotations

import json
import sys
import unittest

sys.path.insert(0, "/Users/sailesh/performance-marketing-agent")

from adops import policies, runtime, scenarios as SC, tools
from adops.agent import SYSTEM_PROMPT, Decision, decide, extract_json, validate_decision
from adops.env import AdAccountEnv
from adops.llm import LLMError, ScriptedBackend


def env(key="S1_creative_fatigue", seed=5):
    return AdAccountEnv(SC.get(key), seed=seed)


ALT = {"level": "creative", "summary": "pause a specific creative instead",
      "why_rejected": "the aggregate ad-set evidence is what supports this action"}
PAT = {"similar_past_action": "none", "reasoning": "no prior action of this shape"}
ACTION_TOOLS = ("set_budget", "set_status", "launch_creative_test")


def j(tool, args=None, decision="act", obs="x"):
    args = dict(args or {})
    if tool in ACTION_TOOLS:
        args.setdefault("considered_alternative", ALT)
        args.setdefault("pattern_check", PAT)
    return json.dumps({"observation": obs, "hypotheses": [{"cause": "c", "evidence": "e",
                                                           "confidence": 0.7}],
                       "decision": decision, "tool": tool, "args": args})


class TestParsing(unittest.TestCase):
    def test_plain_object(self):
        self.assertEqual(extract_json('{"tool":"finish"}')["tool"], "finish")

    def test_fenced_with_prose(self):
        text = 'Sure, here it is:\n```json\n{"tool":"advance_period","args":{}}\n```\nDone.'
        self.assertEqual(extract_json(text)["tool"], "advance_period")

    def test_braces_inside_strings(self):
        text = '{"observation":"a {nested} brace","tool":"finish","args":{"reason":"}"}}'
        self.assertEqual(extract_json(text)["args"]["reason"], "}")

    def test_no_json_raises(self):
        with self.assertRaises(ValueError):
            extract_json("I cannot help with that.")

    def test_missing_tool_rejected(self):
        with self.assertRaises(ValueError):
            validate_decision({"observation": "hi"})


class TestRetry(unittest.TestCase):
    def test_malformed_then_valid(self):
        backend = ScriptedBackend(["not json at all", j("advance_period", {"note": "n"})])
        d = decide(backend, {"period": 1}, [], [])
        self.assertEqual(d.tool, "advance_period")
        self.assertEqual(d.parse_attempts, 2)
        self.assertIn("COULD NOT BE PARSED", backend.prompts[1][1])

    def test_gives_up_after_max_retries(self):
        backend = ScriptedBackend(["nope"] * 3)
        with self.assertRaises(LLMError):
            decide(backend, {"period": 1}, [], [], max_retries=3)


class TestSystemPrompt(unittest.TestCase):
    def test_documents_every_tool(self):
        for name in tools.REGISTRY:
            self.assertIn(name, SYSTEM_PROMPT)

    def test_states_the_cost_of_acting_and_that_waiting_is_allowed(self):
        self.assertIn("learning phase", SYSTEM_PROMPT)
        self.assertIn("Waiting is a legitimate decision", SYSTEM_PROMPT)

    def test_states_both_levels_are_given_by_default(self):
        """Post-mortem fix #1: the prompt must not imply creative data requires a
        tool call to see."""
        self.assertIn("ad-set-level table and a creative-level table", SYSTEM_PROMPT)

    def test_requires_considered_alternative_and_pattern_check(self):
        """Post-mortem fixes #2/#3: both deliberation objects must be documented as
        required, and the prompt must not force a particular level to win."""
        self.assertIn("considered_alternative", SYSTEM_PROMPT)
        self.assertIn("pattern_check", SYSTEM_PROMPT)
        self.assertIn("does not mean the alternative has to lose", SYSTEM_PROMPT)
        self.assertIn("does not forbid repeating an action type", SYSTEM_PROMPT)


class TestAgentRuntime(unittest.TestCase):
    def run_with(self, responses, key="S1_creative_fatigue", seed=5, max_steps=60):
        e = env(key, seed)
        backend = ScriptedBackend(responses)
        pol = policies.LLMPolicy(backend)
        return e, backend, runtime.run_episode(e, pol, max_steps=max_steps)

    def test_invalid_tool_becomes_an_observation_not_a_crash(self):
        e, backend, log = self.run_with(
            [j("teleport_budget"), j("advance_period", {"note": "ok"})] +
            [j("advance_period", {"note": "ok"})] * 8)
        self.assertFalse(log.steps[0].ok)
        self.assertEqual(log.steps[0].error_code, "unknown_tool")
        self.assertIsNone(log.error)
        self.assertIn("REJECTED ACTIONS", backend.prompts[1][1])

    def test_guardrail_rejection_is_fed_back_to_the_model(self):
        bad = j("set_budget", {"entity_id": "as_lal", "new_daily_budget": 999,
                               "rationale": "r",
                               "expected_effect": {"entity_id": "as_lal", "metric": "cpa",
                                                   "direction": "decrease",
                                                   "magnitude_pct": 10, "periods": 2}})
        # (considered_alternative / pattern_check are auto-filled by j() so this
        # rejection isolates the budget-step guardrail, not the deliberation one --
        # see TestDeliberationGuardrails below for those checks in isolation)
        e, backend, log = self.run_with([bad] + [j("advance_period", {"note": "n"})] * 8)
        self.assertEqual(log.steps[0].error_code, "budget_step_too_large")
        self.assertIn("budget_step_too_large", backend.prompts[1][1])
        self.assertEqual(e.account.ad_sets["as_lal"].daily_budget, 95.0)   # unchanged

    def test_runtime_forces_advance_after_repeated_rejections(self):
        bad = j("set_budget", {"entity_id": "nope", "new_daily_budget": 1,
                               "rationale": "r",
                               "expected_effect": {"entity_id": "nope", "metric": "cpa",
                                                   "direction": "decrease",
                                                   "magnitude_pct": 10, "periods": 2}})
        e, backend, log = self.run_with([bad] * 3 + [j("advance_period", {"note": "n"})] * 20)
        forced = [s for s in log.steps if s.forced]
        self.assertTrue(forced, "runtime should force an advance after repeated rejections")
        self.assertEqual(forced[0].tool, "advance_period")

    def test_finish_stops_the_agent_but_not_the_horizon(self):
        e, backend, log = self.run_with([j("finish", {"reason": "done", "summary": "s"},
                                           decision="stop")])
        self.assertEqual(log.steps[-1].tool, "finish")
        self.assertEqual(e.live_period, e.objective.horizon_periods)
        self.assertEqual(log.score["periods_run"], e.objective.horizon_periods)

    def test_model_call_budget_is_capped(self):
        e = env()
        backend = ScriptedBackend([j("advance_period", {"note": "n"})] * 50)
        pol = policies.LLMPolicy(backend, max_llm_calls=2)
        runtime.run_episode(e, pol)
        self.assertLessEqual(backend.calls, 2)

    def test_action_is_recorded_in_the_ledger_with_expected_effect(self):
        act = j("set_status", {"entity_id": "cr_broad_a", "status": "paused",
                               "rationale": "worn out",
                               "expected_effect": {"entity_id": "as_broad", "metric": "cpa",
                                                   "direction": "decrease",
                                                   "magnitude_pct": 15, "periods": 2}})
        e, backend, log = self.run_with([act] + [j("advance_period", {"note": "n"})] * 8)
        self.assertEqual(len(log.ledger), 1)
        entry = log.ledger[0]
        self.assertEqual(entry["expected"]["magnitude_pct"], 15)
        self.assertIn(entry["verdict"], ("confirmed", "partial", "contradicted",
                                         "unmeasurable"))
        self.assertEqual(e.account.creatives["cr_broad_a"].status, "paused")

    def test_prompt_never_contains_ground_truth(self):
        e, backend, log = self.run_with([j("get_account_snapshot", {})] +
                                        [j("advance_period", {"note": "n"})] * 8)
        blob = json.dumps(backend.prompts)
        for key in ("base_cvr", "fatigue_rate", "audience_size", "efficient_weekly_spend",
                    "correct_action_family", "saturation_strength"):
            self.assertNotIn(key, blob)

    def test_scenario_identity_is_not_revealed_to_the_model(self):
        e, backend, log = self.run_with([j("advance_period", {"note": "n"})] * 8)
        blob = json.dumps(backend.prompts)
        for key in SC.keys():
            self.assertNotIn(key, blob)
        self.assertNotIn("fatigue", blob.lower())


class TestDeliberationGuardrailsEndToEnd(unittest.TestCase):
    """Post-mortem fixes #2/#3, exercised through the full LLM policy + runtime, not
    just guardrails.check() in isolation (see TestGuardrails in test_core.py)."""

    def run_with(self, responses, key="S1_creative_fatigue", seed=5, max_steps=60):
        e = env(key, seed)
        backend = ScriptedBackend(responses)
        pol = policies.LLMPolicy(backend)
        return e, backend, runtime.run_episode(e, pol, max_steps=max_steps)

    def test_missing_considered_alternative_is_rejected_and_action_not_applied(self):
        args = {"entity_id": "as_lal", "new_daily_budget": 100, "rationale": "r",
                "expected_effect": {"entity_id": "as_lal", "metric": "cpa",
                                    "direction": "decrease", "magnitude_pct": 10,
                                    "periods": 2},
                "pattern_check": PAT}   # considered_alternative deliberately omitted
        bad = json.dumps({"observation": "o", "hypotheses": [], "decision": "act",
                          "tool": "set_budget", "args": args})
        e, backend, log = self.run_with([bad] + [j("advance_period", {"note": "n"})] * 8)
        self.assertEqual(log.steps[0].error_code, "alternative_not_considered")
        self.assertEqual(e.account.ad_sets["as_lal"].daily_budget, 95.0)

    def test_missing_pattern_check_is_rejected_and_action_not_applied(self):
        args = {"entity_id": "as_lal", "new_daily_budget": 100, "rationale": "r",
                "expected_effect": {"entity_id": "as_lal", "metric": "cpa",
                                    "direction": "decrease", "magnitude_pct": 10,
                                    "periods": 2},
                "considered_alternative": ALT}   # pattern_check deliberately omitted
        bad = json.dumps({"observation": "o", "hypotheses": [], "decision": "act",
                          "tool": "set_budget", "args": args})
        e, backend, log = self.run_with([bad] + [j("advance_period", {"note": "n"})] * 8)
        self.assertEqual(log.steps[0].error_code, "pattern_check_missing")
        self.assertEqual(e.account.ad_sets["as_lal"].daily_budget, 95.0)

    def test_deliberation_does_not_force_a_particular_level(self):
        """A creative-level action with a plausible ad_set-level alternative must be
        just as acceptable as the reverse -- the guardrail checks the field exists
        and is non-empty, never which level was chosen."""
        act = j("set_status", {"entity_id": "cr_broad_a", "status": "paused",
                               "rationale": "r",
                               "expected_effect": {"entity_id": "as_broad", "metric": "cpa",
                                                   "direction": "decrease",
                                                   "magnitude_pct": 10, "periods": 2}})
        e, backend, log = self.run_with([act] + [j("advance_period", {"note": "n"})] * 8)
        self.assertTrue(log.steps[0].ok, log.steps[0].error)

    def test_contradicted_action_becomes_prominently_salient_in_the_next_prompt(self):
        """Post-mortem fix #3: a CONTRADICTED verdict must be hard to miss in the
        very next decision, not just present somewhere in a large JSON blob."""
        cut = j("set_budget", {"entity_id": "as_lal", "new_daily_budget":
                               round(95.0 * 0.8, 2), "rationale": "r",
                               "expected_effect": {"entity_id": "as_lal", "metric": "spend",
                                                   "direction": "increase",
                                                   "magnitude_pct": 10, "periods": 1}})
        e, backend, log = self.run_with(
            [cut, j("advance_period", {"note": "n"})] +
            [j("advance_period", {"note": "n"})] * 8, seed=1)
        # prompts[0] = before the cut, prompts[1] = right after (verdict still
        # pending), prompts[2] = one period later, once the effect has resolved
        self.assertGreaterEqual(len(backend.prompts), 3)
        third_prompt = backend.prompts[2][1]
        self.assertIn("PRIOR ACTIONS AND THEIR MEASURED OUTCOMES", third_prompt)
        self.assertIn("CONTRADICTED", third_prompt)
        self.assertIn("did NOT confirm as expected", third_prompt)


class TestPolicyParity(unittest.TestCase):
    """Every policy must go through the same tool layer and guardrails."""

    def test_noop_makes_no_changes_and_runs_full_horizon(self):
        e = env()
        log = runtime.run_episode(e, policies.NoopPolicy())
        self.assertEqual(log.score["interventions"], 0)
        self.assertEqual(log.score["periods_run"], e.objective.horizon_periods)

    def test_rules_respects_the_change_budget(self):
        for seed in range(1, 5):
            e = env(seed=seed)
            log = runtime.run_episode(e, policies.RulePolicy())
            self.assertLessEqual(log.score["interventions"], e.limits.max_total_changes)
            self.assertEqual(log.score["hard_violations"].count("total_budget_exceeded"), 0)

    def test_no_policy_can_exceed_the_budget_cap(self):
        for key in SC.keys():
            for pol in (policies.NoopPolicy(), policies.RulePolicy(),
                        policies.OraclePolicy(key)):
                e = AdAccountEnv(SC.get(key), seed=2)
                log = runtime.run_episode(e, pol)
                self.assertLessEqual(log.score["spend"], e.objective.total_budget + 1.0,
                                     "%s / %s overspent" % (key, pol.name))


class TestEnvironmentValidity(unittest.TestCase):
    """If a known-correct intervention cannot beat doing nothing, the scenario does
    not reward diagnosis and every other result on it is meaningless."""

    def mean_score(self, key, policy_factory, seeds=range(1, 9)):
        total = 0.0
        for s in seeds:
            e = AdAccountEnv(SC.get(key), seed=s)
            total += runtime.run_episode(e, policy_factory(key)).score["objective_score"]
        return total / len(list(seeds))

    def test_oracle_beats_noop_where_a_real_defect_exists(self):
        for key in ("S1_creative_fatigue", "S2_audience_saturation", "S3_misallocation"):
            noop = self.mean_score(key, lambda k: policies.NoopPolicy())
            oracle = self.mean_score(key, lambda k: policies.OraclePolicy(k))
            self.assertGreater(oracle, noop * 1.05,
                               "%s: oracle %.1f vs noop %.1f -- scenario does not reward "
                               "correct action" % (key, oracle, noop))

    def test_noise_trap_punishes_intervention(self):
        key = "S4_noise_trap"
        noop = self.mean_score(key, lambda k: policies.NoopPolicy())
        rules = self.mean_score(key, lambda k: policies.RulePolicy())
        self.assertGreater(noop, rules,
                           "S4 must punish unnecessary intervention (noop %.1f vs rules %.1f)"
                           % (noop, rules))


if __name__ == "__main__":
    unittest.main(verbosity=2)

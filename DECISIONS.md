# DECISIONS (with rejected alternatives)

D1  Simulated environment, not Meta/Google API.
    Why: an autonomous agent cannot be A/B tested on live spend; a simulator gives a
    repeatable counterfactual. Tool layer is an interface -> swapping in a real API is a
    driver change, not an architecture change.
    Rejected: read-only real API (no closed loop, no feedback).

D2  Hidden ground truth + noisy/delayed observations + cost-to-act.
    Why: without this the simulator rewards any action and the project is self-fulfilling.
    Rejected: deterministic simulator (agent would trivially win).

D3  Single agent.
    Why: no decomposition here reduces context pressure or enables parallelism; multi-agent
    would add coordination failure modes and multiply LLM calls (= Pro usage) for no gain.
    Rejected: analyst/strategist/executor split (theatre).

D4  Deterministic orchestrator owns the loop; LLM owns only decisions.
    Rejected: LLM-driven control flow with self-enforced limits (unsafe, unmeasurable).

D5  Guardrails outside the model, reject-and-explain.
    Invalid actions are rejected deterministically and the reason is returned as an
    observation so the agent can re-plan. Model output can never bypass them.

D6  All three policies (noop / rules / LLM) share ONE runtime and tool layer.
    Why: makes the comparison exact -- only the decision function differs.

D7  Agent must state an expected effect before any consequential action; harness measures
    realised effect deterministically. LLM prediction is never treated as truth.
    Gives calibration metric + revert trigger + honest "did it work" answer.

D8  Briefing is descriptive-only (deltas, trends, significance, sample sizes). No causal
    labels like "fatigue detected" -- otherwise the LLM would just be reading a rules engine.

D9  Stdlib-only Python, target 3.9.6 (system python, no Homebrew/uv on this machine).
    Why: zero install risk, zero cost. Rejected: pandas/numpy/scipy.

D10 Runtime LLM = `claude -p` headless on Pro subscription.
    Feasibility tested: CLI installs free; requires one-time interactive `claude setup-token`.
    No API billing. Full eval matrix runs on deterministic policies; LLM policy runs on a
    subset to respect Pro usage limits.
    Rejected: Anthropic API key (requires billing), local ollama (multi-GB, weak tool JSON).

D11 Four scenarios incl. a NO-REAL-PROBLEM noise trap.
    Why: the discriminating test of a good agent is restraint. Every scenario's correct
    action is another scenario's wrong action.

D12 Post-mortem fixes (S1_creative_fatigue seed 1, see prior turn's diagnosis):
    (a) Creative-level context is now ALWAYS in the default briefing (brief.py
        creative_table), not only reachable via a tool call -- fixes the structural
        over-weighting of ad-set-level state that caused the agent to never once
        call get_account_snapshot or compare_segments(level="creative") on the
        defective ad set.
    (b) Every consequential action requires a `considered_alternative` object
        (the plausible action at the OTHER hierarchy level, and why it lost) and a
        `pattern_check` object (whether this resembles a past action and, if so, why
        acting again is justified given its measured verdict). Both are guardrail-
        enforced -- missing or empty is a deterministic rejection fed back as an
        observation -- exactly like expected_effect. Neither forces a particular
        level or forbids a repeat; they force the deliberation to be stated.
        Applies uniformly to every policy (rules/oracle/llm), not just the LLM, to
        preserve D6 (one runtime, one tool layer, apples-to-apples comparison).
    (c) The user prompt now carries a prose "PRIOR ACTIONS AND THEIR MEASURED
        OUTCOMES" section ahead of the raw JSON state (which is alphabetized by
        `sort_keys=True` and would otherwise bury this), naming CONTRADICTED/PARTIAL
        entries explicitly.
    (d) `evaluate.py`'s diagnosis_hit was fixed: it used to count any action that
        touched an entity in `on_target_entities`, so a plain ad-set budget cut on
        as_broad counted as "diagnosing" S1's creative-fatigue defect. The new
        `mechanism_diagnosis()` requires BOTH investigation at the scenario's
        authored `mechanism_level` (creative/ad_set/none) AND an action matching the
        correct tool KIND and DIRECTION from `correct_action_family` (parsed via a
        small DSL: pause_creative / creative_test / cut_budget / raise_budget /
        shift_budget / none). Re-scoring the actual saved S1 seed-1 trace offline
        (no new LLM call) confirms it now correctly reports a miss.
    Known limitation: OraclePolicy scores investigated_correct_level=False on
    S1-S3 by construction (it applies a pre-known fix without investigating), so
    its diagnosis_hit is also False even though acted_on_mechanism is always True.
    This is intentional and documented, not a bug -- see PROJECT_STATE.md.

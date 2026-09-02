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

# AdOps Agent

A closed-loop agent that operates a **simulated** performance-marketing account: it
observes noisy state, forms hypotheses about *why* performance diverges from a business
objective, acts through a validated tool layer, observes the consequences, and decides
whether to act again or stop.

Python 3.9+, **standard library only**. No paid API, no database, no server.

## Why a simulator
An autonomous agent cannot be A/B tested on live ad spend, and a real account gives no
counterfactual. The simulator holds **hidden ground truth** the agent can never inspect,
so the same scenario can be replayed against different policies and scored. The tool layer
is an interface: swapping in a real Marketing API is a driver change, not an
architecture change.

## Quick start
```bash
python3 -m unittest discover -s tests -q          # 98 tests
python3 -m adops sim --scenario S1_creative_fatigue --seed 1
python3 -m adops run --scenario S1_creative_fatigue --seed 1 --policy rules --report
python3 -m adops eval --policies noop,rules,oracle --seeds 5
```

## The loop
```
OBSERVE   noisy account state, delayed conversions      [code]
ORIENT    deltas, trends, significance, sufficiency     [code]  brief.py
REASON    hypotheses -> investigate / act / wait / stop [policy] agent.py
VALIDATE  guardrails; a rejection is fed back as an observation  guardrails.py
ACT       one tool call mutates the environment         [code]  tools.py
ADVANCE   one simulated period elapses                  [code]  env.py
EVALUATE  realised effect vs the agent's stated expectation      env.py
```
The model decides *what to do*. The harness decides *what is allowed* and computes
*every number*. The LLM never performs arithmetic, never enforces a constraint and never
sees hidden state (there are tests for both).

## Environment
Seven ad sets across three campaigns, fourteen creatives, twelve periods of history,
a six-period horizon. Hidden dynamics:

- latent conversion rate, CPM and audience size per ad set
- two independent diminishing-return mechanisms: **capacity push** (CPM inflation and
  audience-quality dilution past an efficient weekly spend) and **audience saturation**
  (frequency-driven, on small audiences) -- they have different observable signatures
- **creative fatigue** with configurable persistence
- **learning phase**: any structural change costs efficiency for the next period
- binomial sampling noise, and **25% of conversions reported one period late**

### Scenarios
| key | hidden cause | correct response |
|---|---|---|
| `S1_creative_fatigue` | a worn-out hero creative in the largest ad set | pause it / test a new one |
| `S2_audience_saturation` | retargeting scaled past what a small audience absorbs | cut its budget |
| `S3_misallocation` | budget concentrated on the structurally weaker ad set | shift budget |
| `S4_noise_trap` | **nothing** -- every latent parameter is healthy | do nothing |

`S4` is the discriminating test: acting there always costs a learning penalty for no gain.

## Tools
Read: `get_account_snapshot`, `get_timeseries`, `compare_segments` (returns significance
verdicts, so the agent never eyeballs whether a difference is real).
Act: `set_budget`, `set_status`, `launch_creative_test`.
Control: `advance_period` (also the way to wait), `finish`.

Every action requires a rationale and an **expected effect** (entity, metric, direction,
magnitude, horizon). The harness measures the realised effect deterministically and
records confirmed / partial / contradicted. The model's prediction is never treated as
ground truth.

## Guardrails (outside the model)
Per-action budget step cap, per-ad-set and per-account budget ceilings, campaign spend
floors, a **pacing guard** that refuses an increase the budget cannot fund, minimum data
before pausing or cutting, minimum active assets, per-period and total change budgets,
tool-call budgets, and an action whitelist. A rejection returns a machine-readable reason
that is fed back to the agent as an observation.

## Policies
`noop` (never intervene) - `rules` (a reasonable deterministic auto-optimiser with
significance checks and cooldowns) - `llm` (the agent) - `oracle` (evaluation only:
applies the known-correct fix, and exists to prove the environment rewards correct
diagnosis at all).

## Evaluation
`python3 -m adops eval` runs a seeded matrix and reports conversions, CPA, objective
score, lift vs do-nothing, **regret vs oracle**, action counts, diagnosis-hit rate,
unnecessary actions, constraint violations and prediction accuracy. Same seed, same
result, across processes.

## Runtime model
The agent depends on `LLMBackend`, not on a provider. `ClaudeCLIBackend` shells out to
the Claude Code CLI, which authenticates against an existing Claude subscription:

```bash
npm install -g @anthropic-ai/claude-code   # free
claude setup-token                         # one time, uses your subscription
python3 -m adops probe-llm
```
No Anthropic API key, no billing, no paid service is used anywhere in this project. If no
model is available the system still runs end to end on the deterministic policies, and
`ScriptedBackend` exercises the whole agent runtime in tests with no model at all.

# PROJECT_STATE

Updated: after STEP 11. Everything works except a real-model agent run,
which is blocked on a one-time `claude setup-token` login.

## Done
- Repo memory files, package skeleton, git repo.
- `metrics.py`, `stats.py` -- all deterministic arithmetic and significance testing.
- `world.py` -- hidden ground truth. Two independent diminishing-return mechanisms:
  capacity push (CPM inflation + CVR dilution past `efficient_weekly_spend`) and
  audience saturation (frequency-driven, small-audience). Creative fatigue with a
  configurable persistence (`decay`). Learning-phase multiplier. Binomial sampling.
- `scenarios.py` -- S0 calibration + S1 fatigue / S2 saturation / S3 misallocation /
  S4 noise trap. Budgets equilibrated by hill-climb so no generic reallocation
  dominates; each defect costs 12-24% CPA vs the healthy account.
- `env.py` -- observable state, 12-period burn-in history, period advance, action
  application, reporting delay (25% of conversions land one period late), action
  ledger with expected-vs-realised scoring, deterministic objective scoring.
- `guardrails.py` -- 15 hard rules incl. pacing guard, step caps, floors, data
  minimums, change budgets. Rejections carry machine-readable codes.
- `tools.py` -- 8 tools (3 read / 3 action / 2 control), uniform envelope.
- `brief.py` -- descriptive-only ORIENT briefing.
- `tests/test_core.py` -- 33 tests, all passing, incl. hidden-state-leak test and
  scenario-signature tests.

## Calibrated numbers (do-nothing baseline, 15 seeds)
| scenario | conv | CPA | target_cpa | budget |
|---|---|---|---|---|
| S1_creative_fatigue | ~500 | 40.3 | 37.0 | 20563 |
| S2_audience_saturation | ~500 | 44.5 | 41.0 | 22705 |
| S3_misallocation | ~490 | 42.4 | 39.0 | 21206 |
| S4_noise_trap | ~569 | 35.5 | 38.0 | 20563 (target already met) |

## Gotcha found and fixed
`hash()` on str is salted per process -> seeds were not reproducible across runs.
Now uses `zlib.crc32`. Never reintroduce `hash()` for seeding.

## Baseline results (6 seeds, deterministic, reproducible)
objective_score, lift vs do-nothing:
| scenario | noop | rules | oracle |
|---|---|---|---|
| S1_creative_fatigue | 519.1 | 495.8 (-4.5%) | 596.8 (+15.0%) |
| S2_audience_saturation | 459.7 | 487.4 (+6.0%) | 508.3 (+10.6%) |
| S3_misallocation | 446.8 | 442.3 (-1.0%) | 481.6 (+7.8%) |
| S4_noise_trap | 551.8 | 507.0 (-8.1%) | 551.8 (0.0%) |

The oracle beating noop on S1-S3 is the environment-validity check: the scenarios do
reward correct diagnosis. The rules baseline loses because it chases noisy CPA outliers
and re-scales retargeting into saturation -- a real failure mode, not a strawman.
There is therefore genuine headroom between `rules` and `oracle` for the agent to occupy.

## Post-mortem fixes applied (fixes #1-4, prior turn)
Creative context in the default briefing, guardrail-enforced considered_alternative /
pattern_check on every consequential action, prose salience of contradicted/partial
outcomes, and a general mechanism-level diagnosis metric. See DECISIONS.md D12.

Result: the fixes visibly changed agent behaviour (investigated the creative level
immediately, named the correct hypothesis, generalised across contradicted verdicts
to stop acting) but the agent still did not act on the creative level -- see failure
analysis below.

## Failure analysis + statistical-power fix (this turn)
Root cause: the agent required CVR/CPA statistical significance before it would
consider a creative-level action, but CVR/CPA is structurally underpowered relative
to CTR/CPC at equal sample size in this environment (and in real ad accounts).
Fix: one scenario-agnostic statistical-power paragraph added to SYSTEM_PROMPT (see
DECISIONS.md D13). Prompt-only change -- no simulator, tool, guardrail, action-
semantics, objective, evaluation, or runtime change.

98 deterministic tests pass. Deterministic policies (noop/rules/oracle) verified
byte-identical to pre-change scores (they never read SYSTEM_PROMPT). 3 new tests
lock in that the guidance is present, does not force a level or action, and names
no scenario or entity.

A second contributing factor (the 20%-threshold learning-phase cost asymmetry makes
sub-threshold budget nudges the only zero-cost action) was identified but
deliberately NOT addressed -- flagged as a separate, deferred lever.

## Next
Awaiting approval for a controlled S1_creative_fatigue seed-1 LLM re-run to test
whether the statistical-power guidance changes acted_on_mechanism / diagnosis_hit,
per the evaluation protocol proposed in the failure analysis (multi-seed, plus a
placebo check on S2/S3/S4 to rule out an unwanted creative-eagerness bias).

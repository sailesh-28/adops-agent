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

## Post-mortem fixes applied (this turn)
Implemented all 4 fixes from the S1_creative_fatigue seed-1 post-mortem: creative
context in the default briefing, guardrail-enforced considered_alternative /
pattern_check on every consequential action (all policies, not just LLM), prose
salience of contradicted/partial outcomes in the prompt, and a general mechanism-
level diagnosis metric (`evaluate.mechanism_diagnosis`) replacing the old
entity-touch-only `diagnosis_hit`. Full details in DECISIONS.md D12.

95 deterministic tests pass (`python3 -m unittest discover -s tests -q`), including
a regression test that re-scores the actual saved S1 seed-1 LLM trace from the
post-mortem and confirms the fixed metric now reports it as a miss.

No LLM evaluation has been run since the fixes landed -- awaiting approval.

## Known/documented limitation
OraclePolicy never calls a read tool (it applies a pre-known fix), so
`investigated_correct_level` is False for it on S1-S3 and its `diagnosis_hit` is
therefore also False despite `acted_on_mechanism` always being True. This is
correct behaviour for a metric that is honest about what "diagnosis" means, not a
bug -- `acted_on_mechanism_rate` is the number to read for the oracle specifically.

## Next
- Await approval, then run `python3 -m adops run --policy llm --report` on
  S1_creative_fatigue seed 1 again and compare against this turn's fixes.
- If the creative-level blind spot persists, the next lever is prompt emphasis
  (fix #1 made the data available; it did not force the model to look).

# PROJECT_STATE

Updated: after STEP 5 (simulator + tools + guardrails tested).

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

## Next
STEP 6 agent + LLM backend abstraction -> STEP 7 run -> STEP 8 baselines ->
STEP 9 evaluation -> STEP 10 CLI -> STEP 11 report.

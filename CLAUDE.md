# CLAUDE.md -- working rules for this repo

- Python 3.9.6, **stdlib only**. No pip installs. `from __future__ import annotations` in
  every module so PEP-585 annotations work.
- The LLM never does arithmetic, never enforces a constraint, never sees hidden state.
  If you are tempted to have the model compute a number, put it in `metrics.py`/`stats.py`.
- Guardrails live in `guardrails.py`, outside the model. Rejections return a reason that is
  fed back to the agent as an observation.
- `world.py` is hidden ground truth. Nothing in `tools.py` may read it directly.
- Determinism: every run is seeded. Same (scenario, seed, policy) => identical result.
  Never use the global `random` module; always the injected `random.Random` instance.
- `brief.py` output must stay descriptive. No causal labels.
- Run tests with: `python3 -m unittest discover -s tests -q`
- Read PROJECT_STATE.md for where things stand; update it at milestones. Keep docs terse.

# ARCHITECTURE

## Layers (strictly separated)
```
scenarios.py   scenario configs -> hidden params + account structure + objective
world.py       HIDDEN GROUND TRUTH + stochastic dynamics   (agent can NEVER read this)
env.py         AdAccountEnv: observable state, action application, period advance, scoring
guardrails.py  deterministic validation of every proposed action (outside the LLM)
tools.py       tool registry: schemas, dispatch, uniform result envelope
metrics.py     all derived metrics + aggregation (no LLM arithmetic, ever)
stats.py       significance tests, sufficiency verdicts, trend estimation
brief.py       ORIENT: deterministic evidence briefing (descriptive only, no causal labels)
agent.py       the loop; prompt assembly; JSON decision parsing; retry; effect ledger
llm.py         LLMBackend abstraction (ClaudeCLI / Scripted / Unavailable)
policies.py    NoopPolicy, RulePolicy, LLMPolicy  -- all share one runtime + tool layer
evaluate.py    seeded matrix: policies x scenarios x seeds -> deterministic metrics
cli.py         python3 -m adops <probe-llm|sim|run|eval|report>
```

## The loop
```
OBSERVE    env.observation()                       [code]
ORIENT     brief.build() deltas/trends/sig tests   [code]
REASON     policy.decide() -> Decision(JSON)       [LLM or rules]
VALIDATE   guardrails.check()  reject -> feedback  [code]
ACT        tools.dispatch() mutates env            [code]
ADVANCE    env.advance() one simulated period      [code]
EVALUATE   expected vs realised effect, ledger     [code]
DECIDE     continue / stop                         [LLM] + hard caps [code]
```

## Control split
The model decides *what to do*. The harness decides *what is allowed* and computes *every
number*. The LLM never performs arithmetic, never enforces a constraint, never touches
hidden state.

## Prompt/context discipline
Prompt = static (objective, constraints, tool schemas) + rolling window (last K tool
results) + full action ledger + current deterministic briefing. Bounded size; does not grow
with run length.

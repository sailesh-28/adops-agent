# PROJECT_BRIEF

## What this is
A closed-loop agent that operates a **simulated** performance-marketing account. It observes
noisy account state, forms hypotheses about *why* performance diverges from a business
objective, takes permitted actions through a validated tool layer, observes consequences,
and decides whether to continue or stop.

## Business problem
Ad platforms report *what* happened. They do not say *why* CPA moved, *which* of several
plausible causes it is, *what* to do, or *whether the fix worked*. That diagnostic loop is
manual, slow and inconsistent. Wrong diagnosis costs money twice: the wasted spend, plus the
learning-phase reset caused by the unnecessary edit.

## Objective the agent optimises
Maximise conversions over a 6-period horizon subject to:
- hard: blended CPA <= target; cumulative spend <= budget cap; campaign floor spend respected
- soft: weekly pacing within tolerance; limited number of structural changes
- penalty: every structural edit triggers a learning-phase efficiency hit next period

## Non-goals
No real ad-platform integration. No dashboard. No DB. No multi-agent theatre. No paid API.

## Zero-cost constraint
No billing, no API credits, no paid services. Runtime LLM = Claude Code CLI headless
(`claude -p`) on the existing Pro subscription. System is fully functional without any LLM
(deterministic policies).

## Key design commitment
The simulator holds **hidden ground truth** the agent can never inspect. Observations are
noisy, partially delayed, and acting has a cost. The agent must infer cause from incomplete
evidence, and sometimes the correct decision is to do nothing.

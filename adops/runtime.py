"""The deterministic orchestrator.

Owns the loop, the budgets, the failsafes and the audit log. Every policy --
do-nothing, rules, LLM -- runs through this identical runtime and the identical
tool layer, so the comparison between them is exact.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import List, Optional

from . import brief as B
from . import tools
from .agent import Decision

RECENT_WINDOW = 4
MAX_CONSECUTIVE_REJECTIONS = 3


@dataclass
class StepLog:
    step: int
    period: int
    tool: str
    args: dict
    ok: bool
    decision: str = ""
    observation: str = ""
    hypotheses: List[dict] = field(default_factory=list)
    error_code: Optional[str] = None
    error: Optional[str] = None
    forced: bool = False
    parse_attempts: int = 1
    llm_meta: dict = field(default_factory=dict)


@dataclass
class RunLog:
    scenario: str
    seed: int
    policy: str
    steps: List[StepLog] = field(default_factory=list)
    score: dict = field(default_factory=dict)
    ledger: List[dict] = field(default_factory=list)
    rejections: List[dict] = field(default_factory=list)
    wall_seconds: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def run_episode(env, policy, max_steps: int = 160, verbose: bool = False) -> RunLog:
    log = RunLog(scenario=env.scenario.key, seed=env.seed, policy=policy.name)
    recent: List[dict] = []
    seen_rejections = 0
    consecutive_rejections = 0
    started = time.time()
    step = 0

    try:
        while not env.is_done() and step < max_steps:
            step += 1
            briefing = B.build(env)
            new_rejections = env.rejections[seen_rejections:]
            seen_rejections = len(env.rejections)

            note = ""
            forced = False
            if consecutive_rejections >= MAX_CONSECUTIVE_REJECTIONS:
                decision = Decision(tool="advance_period",
                                    args={"note": "forced by runtime after repeated "
                                                  "rejected actions"},
                                    decision="wait")
                forced = True
                consecutive_rejections = 0
            else:
                decision = policy.decide(env, briefing, recent, new_rejections, note)

            result = tools.dispatch(env, decision.tool, decision.args)
            recent.append({"tool": decision.tool, "args": decision.args,
                           "result": result.get("data") if result["ok"]
                           else {"error": result.get("error"),
                                 "error_code": result.get("error_code")}})
            recent = recent[-RECENT_WINDOW:]

            log.steps.append(StepLog(
                step=step, period=env.live_period, tool=decision.tool, args=decision.args,
                ok=result["ok"], decision=decision.decision, observation=decision.observation,
                hypotheses=decision.hypotheses, error_code=result.get("error_code"),
                error=result.get("error"), forced=forced,
                parse_attempts=getattr(decision, "parse_attempts", 1),
                llm_meta=getattr(decision, "meta", {}) or {}))

            if verbose:
                mark = "ok " if result["ok"] else "REJ"
                print("  [%02d] p%d %s %-22s %s" % (step, env.live_period, mark,
                                                    decision.tool,
                                                    result.get("error_code") or
                                                    (decision.decision or "")))

            if result["ok"]:
                consecutive_rejections = 0
                if decision.tool == "finish":
                    break
            else:
                consecutive_rejections += 1
    except Exception as exc:                                    # noqa: BLE001
        log.error = "%s: %s" % (type(exc).__name__, exc)

    # An agent that stops early is declining to intervene further -- the account
    # still runs to the end of the horizon. Stopping must not shorten the horizon.
    # NOTE: deliberately ignores `finished_reason` -- calling finish() means "I have
    # no further intervention to make", not "stop the clock".
    while env.periods_remaining() > 0:
        env.advance()
    env.force_score_effects()

    log.score = env.score()
    log.ledger = [e.__dict__ for e in env.ledger]
    log.rejections = env.rejections
    log.wall_seconds = round(time.time() - started, 2)
    return log

"""The LLM decision function: prompt assembly, parsing, validation, retry.

The model chooses WHAT to do. It never computes a metric, never enforces a limit
and never touches the environment directly -- `runtime.py` and `guardrails.py` do that.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .llm import LLMBackend, LLMError
from .tools import (CONSIDERED_ALTERNATIVE_SPEC, EXPECTED_EFFECT_SPEC,
                    PATTERN_CHECK_SPEC, TOOL_SCHEMAS)

MAX_RESULT_CHARS = 2600
RECENT_RESULTS = 4


def _tool_docs() -> str:
    lines = []
    for t in TOOL_SCHEMAS:
        args = "\n".join("      - %s: %s" % (k, v) for k, v in t["args"].items())
        lines.append("  %s  [%s]\n    %s\n    arguments:\n%s"
                     % (t["name"], t["kind"], t["description"], args))
    return "\n".join(lines)


SYSTEM_PROMPT = """You are an autonomous performance-marketing optimisation agent working \
on one advertising account inside a controlled simulation.

YOUR GOAL
Maximise total conversions over the planning horizon while keeping blended CPA at or \
below the target and total spend within the budget cap.

HOW THIS ENVIRONMENT WORKS
- You act by calling exactly one tool per turn. The sequence is yours to choose.
- Structural changes (budget changes of 20%% or more, pausing, launching a creative test) \
put the affected ad set into a learning phase: its efficiency is reduced for the next \
period. Acting therefore has a real cost. An unnecessary change makes things worse.
- Conversions are reported with a delay: roughly a quarter of a period's conversions \
appear in the next period's figures. Judging an action after a single period is unreliable.
- Your change budget is limited. So is the number of tool calls per period.
- Waiting is a legitimate decision. If the evidence does not support a change, advance \
the period without acting.
- Hard guardrails validate every action before it executes. If an action is rejected you \
will see the reason; adapt rather than repeating it.

STATE YOU ARE GIVEN
The account has two levels below campaign: ad sets and, inside each ad set, creatives. \
Every state you receive includes BOTH an ad-set-level table and a creative-level table by \
default -- you never have to call a tool just to see basic creative CTR/CPA. A real \
problem can live at either level, or at neither. Nothing here tells you which; that is \
still your call, and `compare_segments` / `get_timeseries` remain how you get statistical \
tests and history at whichever level you choose to dig into.

WHAT YOU MUST NOT DO
- Do not compute metrics yourself. The tools return computed metrics, confidence \
intervals and statistical verdicts. Use them.
- Do not treat an unexplained difference as a cause. `compare_segments` tells you whether \
a difference is distinguishable from noise; "insufficient_data" means you do not know yet.
- Do not act on a hypothesis you have not checked when you still have tool calls available.

TOOLS
%s

BEFORE A CONSEQUENTIAL ACTION (set_budget, set_status, launch_creative_test)
Three objects are required inside `args`, in addition to the tool's own arguments:

expected_effect (required)
%s

considered_alternative (required) -- the plausible action at the OTHER hierarchy level
%s
This does not mean the alternative has to lose in general -- state it honestly and say \
why the evidence favours your chosen level THIS time. If you truly see no plausible \
alternative at the other level, say that and say why.

pattern_check (required) -- whether this resembles a past action of yours
%s
The state you are given always lists your own past actions and their MEASURED verdicts \
(confirmed / partial / contradicted / unmeasurable), including any considered_alternative \
and pattern_check you stated for them at the time. Before proposing a new action, check \
whether it has the same shape (same tool, same kind of entity, same direction of change) \
as a past action whose verdict was "contradicted" or "partial". This does not forbid \
repeating an action type -- it forbids repeating one without acknowledging what happened \
last time and saying explicitly why this case is different, or why acting again is still \
justified anyway.

OUTPUT CONTRACT
Reply with exactly one JSON object and nothing else. No prose before or after, no code \
fences. Schema:
{
  "observation": "what the current evidence shows, 1-3 sentences",
  "hypotheses": [
    {"cause": "a specific mechanism", "evidence": "the measurement supporting it",
     "confidence": 0.0-1.0}
  ],
  "decision": "investigate" | "act" | "wait" | "stop",
  "tool": "<one tool name>",
  "args": { ... arguments for that tool, including the three objects above when the \
tool is set_budget, set_status or launch_creative_test ... }
}
Rank hypotheses most likely first. If you are acting, the chosen action must follow from \
the top hypothesis and the evidence you have actually gathered.""" % (
    _tool_docs(),
    "\n".join("      - %s: %s" % (k, v) for k, v in EXPECTED_EFFECT_SPEC.items()),
    "\n".join("      - %s: %s" % (k, v) for k, v in CONSIDERED_ALTERNATIVE_SPEC.items()),
    "\n".join("      - %s: %s" % (k, v) for k, v in PATTERN_CHECK_SPEC.items()),
)


@dataclass
class Decision:
    tool: str
    args: dict = field(default_factory=dict)
    observation: str = ""
    hypotheses: List[dict] = field(default_factory=list)
    decision: str = ""
    raw: str = ""
    parse_attempts: int = 1
    meta: dict = field(default_factory=dict)


def extract_json(text: str) -> dict:
    """Pull the first balanced JSON object out of a model response."""
    if text is None:
        raise ValueError("empty response")
    s = text.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s
        s = s.rsplit("```", 1)[0]
    start = s.find("{")
    if start < 0:
        raise ValueError("no JSON object found in response")
    depth, in_str, esc = 0, False, False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(s[start:i + 1])
    raise ValueError("unterminated JSON object in response")


def validate_decision(obj: dict) -> Decision:
    if not isinstance(obj, dict):
        raise ValueError("response is not a JSON object")
    tool = obj.get("tool")
    if not isinstance(tool, str) or not tool:
        raise ValueError("missing 'tool' (a string naming exactly one tool)")
    args = obj.get("args", {})
    if args is None:
        args = {}
    if not isinstance(args, dict):
        raise ValueError("'args' must be an object")
    hyps = obj.get("hypotheses") or []
    if not isinstance(hyps, list):
        hyps = [{"cause": str(hyps)}]
    return Decision(tool=tool, args=args,
                    observation=str(obj.get("observation", ""))[:1200],
                    hypotheses=hyps[:5],
                    decision=str(obj.get("decision", ""))[:40])


def _truncate(payload, limit: int = MAX_RESULT_CHARS) -> str:
    text = json.dumps(payload, sort_keys=True)
    if len(text) <= limit:
        return text
    return text[:limit] + ' ... [truncated: call the tool with a narrower scope]"'


def _fmt_ledger_entry(e: dict) -> str:
    exp = e.get("expected") or {}
    verdict = e.get("verdict", "pending")
    tag = "PENDING" if verdict == "pending" else verdict.upper()
    core = {k: v for k, v in (e.get("args") or {}).items()
           if k in ("entity_id", "new_daily_budget", "status", "ad_set_id")}
    line = "  #%d p%d %s(%s) -> %s" % (e["seq"], e["period"], e["tool"],
                                       json.dumps(core, sort_keys=True), tag)
    if e.get("detail"):
        line += ": %s" % e["detail"]
    alt = e.get("considered_alternative") or {}
    pat = e.get("pattern_check") or {}
    if alt:
        line += "\n      (at the time, alternative considered: %s -- %s)" % (
            alt.get("level"), alt.get("why_rejected", "")[:160])
    if pat and str(pat.get("similar_past_action", "none")) != "none":
        line += "\n      (at the time, flagged as similar to #%s: %s)" % (
            pat.get("similar_past_action"), pat.get("reasoning", "")[:160])
    return line


def summarize_ledger(ledger: List[dict]) -> str:
    """Prose recap of prior actions and their measured outcomes, placed ahead of the
    raw JSON state so it cannot be missed. Deliberately surfaces CONTRADICTED and
    PARTIAL verdicts first -- those are the ones a repeat action must reckon with."""
    if not ledger:
        return ""
    bad = [e for e in ledger if e["verdict"] in ("contradicted", "partial")]
    lines = ["\nYOUR PRIOR ACTIONS AND THEIR MEASURED OUTCOMES (most recent last):"]
    for e in ledger:
        lines.append(_fmt_ledger_entry(e))
    if bad:
        lines.append("\n%d of your past action(s) above did NOT confirm as expected. "
                     "If your next action has the same tool, same kind of entity, and "
                     "same direction of change as one of these, your pattern_check must "
                     "name it and explain why acting again is still justified." % len(bad))
    return "\n".join(lines)


def build_user_prompt(briefing: dict, recent: List[dict], rejections: List[dict],
                      note: str = "") -> str:
    parts = [
        "CURRENT STATE (all figures computed deterministically by the platform):",
        json.dumps(briefing, sort_keys=True, indent=1),
    ]
    ledger_summary = summarize_ledger(briefing.get("action_ledger") or [])
    if ledger_summary:
        parts.append(ledger_summary)
    if recent:
        parts.append("\nRESULTS OF YOUR LAST %d TOOL CALLS (most recent last):" % len(recent))
        for r in recent:
            parts.append("- %s" % _truncate(r))
    if rejections:
        parts.append("\nREJECTED ACTIONS (the guardrail layer refused these; do not repeat "
                     "them unchanged):")
        for r in rejections:
            parts.append("- %s %s -> %s: %s" % (r["tool"], json.dumps(r.get("args", {}),
                                                                      sort_keys=True),
                                                r.get("code"), r.get("message", "")))
    if note:
        parts.append("\nNOTE: %s" % note)
    parts.append("\nDecide your next single tool call. Reply with one JSON object only.")
    return "\n".join(parts)


def decide(backend: LLMBackend, briefing: dict, recent: List[dict],
           rejections: List[dict], note: str = "",
           max_retries: int = 3) -> Decision:
    """Ask the model for one decision. Retries on malformed output, feeding the
    parse error back so the model can correct itself."""
    user = build_user_prompt(briefing, recent, rejections, note)
    last_error = ""
    for attempt in range(1, max_retries + 1):
        prompt = user if attempt == 1 else (
            user + "\n\nYOUR PREVIOUS REPLY COULD NOT BE PARSED: %s\n"
                   "Reply with exactly one JSON object matching the output contract, "
                   "and nothing else." % last_error)
        response = backend.complete(SYSTEM_PROMPT, prompt)
        try:
            decision = validate_decision(extract_json(response.text))
        except (ValueError, TypeError) as exc:
            last_error = str(exc)[:300]
            continue
        decision.raw = response.text[:4000]
        decision.parse_attempts = attempt
        decision.meta = response.meta
        return decision
    raise LLMError("model produced unparseable output %d times: %s"
                   % (max_retries, last_error))

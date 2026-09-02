"""The LLM decision function: prompt assembly, parsing, validation, retry.

The model chooses WHAT to do. It never computes a metric, never enforces a limit
and never touches the environment directly -- `runtime.py` and `guardrails.py` do that.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .llm import LLMBackend, LLMError
from .tools import EXPECTED_EFFECT_SPEC, TOOL_SCHEMAS

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

WHAT YOU MUST NOT DO
- Do not compute metrics yourself. The tools return computed metrics, confidence \
intervals and statistical verdicts. Use them.
- Do not treat an unexplained difference as a cause. `compare_segments` tells you whether \
a difference is distinguishable from noise; "insufficient_data" means you do not know yet.
- Do not act on a hypothesis you have not checked when you still have tool calls available.

TOOLS
%s

EXPECTED-EFFECT OBJECT (required on set_budget, set_status and launch_creative_test)
%s

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
  "args": { ... arguments for that tool ... }
}
Rank hypotheses most likely first. If you are acting, the chosen action must follow from \
the top hypothesis and the evidence you have actually gathered.""" % (
    _tool_docs(),
    "\n".join("      - %s: %s" % (k, v) for k, v in EXPECTED_EFFECT_SPEC.items()),
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


def build_user_prompt(briefing: dict, recent: List[dict], rejections: List[dict],
                      note: str = "") -> str:
    parts = [
        "CURRENT STATE (all figures computed deterministically by the platform):",
        json.dumps(briefing, sort_keys=True, indent=1),
    ]
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

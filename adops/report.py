"""Run reporting: a readable decision narrative plus a self-contained HTML file.

Everything here is presentation over data already computed elsewhere. No analysis
happens in this module.
"""
from __future__ import annotations

import html
import json
import os
from typing import List, Optional

VERDICT_LABEL = {
    "confirmed": "confirmed",
    "partial": "partially confirmed",
    "contradicted": "CONTRADICTED",
    "unmeasurable": "not measurable",
    "pending": "pending",
}


def narrative(log: dict) -> str:
    """Plain-text decision narrative for one run."""
    out: List[str] = []
    score = log.get("score", {})
    out.append("RUN  scenario=%s  seed=%s  policy=%s" % (log["scenario"], log["seed"],
                                                         log["policy"]))
    out.append("=" * 78)
    current = None
    for s in log["steps"]:
        if s["period"] != current:
            current = s["period"]
            out.append("")
            out.append("-- period %d --" % current)
        head = "  %-22s %s" % (s["tool"], "ok" if s["ok"] else "REJECTED: %s"
                               % (s.get("error_code") or ""))
        out.append(head)
        if s.get("observation"):
            out.append("     saw: %s" % s["observation"][:220])
        for h in (s.get("hypotheses") or [])[:3]:
            out.append("     hypothesis: %s (confidence %s)"
                       % (str(h.get("cause"))[:120], h.get("confidence")))
            if h.get("evidence"):
                out.append("        evidence: %s" % str(h["evidence"])[:160])
        if s["tool"] in ("set_budget", "set_status", "launch_creative_test") and s["ok"]:
            out.append("     args: %s" % json.dumps(s["args"], sort_keys=True)[:220])
    out.append("")
    out.append("ACTIONS AND WHETHER THEY WORKED")
    if not log.get("ledger"):
        out.append("  (no structural changes were made)")
    for e in log.get("ledger", []):
        exp = e.get("expected") or {}
        out.append("  #%s p%s %s %s" % (e["seq"], e["period"], e["tool"],
                                        json.dumps(e["args"], sort_keys=True)[:120]))
        out.append("      rationale : %s" % str(e.get("rationale", ""))[:200])
        out.append("      expected  : %s %s %s%% within %s period(s)"
                   % (exp.get("entity_id"), exp.get("metric"), exp.get("direction"),
                      exp.get("magnitude_pct")))
        out.append("      outcome   : %s -- %s" % (VERDICT_LABEL.get(e["verdict"],
                                                                     e["verdict"]),
                                                   e.get("detail", "")))
    out.append("")
    out.append("REJECTED ACTIONS: %d" % len(log.get("rejections", [])))
    for r in log.get("rejections", [])[:12]:
        out.append("  p%s %s -> %s: %s" % (r.get("period"), r.get("tool"),
                                           r.get("code"), str(r.get("message", ""))[:120]))
    out.append("")
    out.append("RESULT")
    for k in ("conversions", "cpa", "target_cpa", "spend", "total_budget",
              "budget_utilisation", "objective_score", "interventions", "tool_calls",
              "hard_violations", "effect_verdicts", "finished_reason"):
        if k in score:
            out.append("  %-20s %s" % (k, score[k]))
    return "\n".join(out)


_CSS = """
:root{--bg:#fff;--fg:#1a1a1a;--mut:#666;--line:#e2e2e2;--ok:#0a7c3f;--bad:#b3261e;
--card:#fafafa}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#131313;--fg:#ececec;
--mut:#9a9a9a;--line:#2e2e2e;--ok:#4ade80;--bad:#ff6b6b;--card:#1c1c1c}}
:root[data-theme=dark]{--bg:#131313;--fg:#ececec;--mut:#9a9a9a;--line:#2e2e2e;
--ok:#4ade80;--bad:#ff6b6b;--card:#1c1c1c}
body{background:var(--bg);color:var(--fg);font:14px/1.5 ui-monospace,SFMono-Regular,
Menlo,monospace;margin:0;padding:24px}
h1{font-size:20px;margin:0 0 4px} h2{font-size:15px;margin:28px 0 8px;
border-bottom:1px solid var(--line);padding-bottom:4px}
.mut{color:var(--mut)} .ok{color:var(--ok)} .bad{color:var(--bad)}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{text-align:right;padding:4px 8px;border-bottom:1px solid var(--line)}
th:first-child,td:first-child{text-align:left}
.wrap{overflow-x:auto}
.step{border-left:2px solid var(--line);padding:2px 0 2px 10px;margin:6px 0}
.card{background:var(--card);border:1px solid var(--line);padding:10px 12px;margin:8px 0}
"""


def _esc(v) -> str:
    return html.escape(str(v))


def _table(headers: List[str], rows: List[List[object]]) -> str:
    head = "".join("<th>%s</th>" % _esc(h) for h in headers)
    body = "".join("<tr>%s</tr>" % "".join("<td>%s</td>" % _esc("-" if c is None else c)
                                           for c in r) for r in rows)
    return '<div class="wrap"><table><tr>%s</tr>%s</table></div>' % (head, body)


def evaluation_html(result: dict, title: str = "Evaluation") -> str:
    parts = ["<style>%s</style>" % _CSS, "<h1>%s</h1>" % _esc(title),
             '<p class="mut">%d scenario(s) &times; %d seed(s) &times; %d polic(ies); '
             'every run seeded and reproducible.</p>'
             % (len(result["scenarios"]), len(result["seeds"]), len(result["policies"]))]
    headers = ["policy", "runs", "conversions", "CPA", "score", "sd", "lift vs noop %",
               "regret vs oracle", "actions", "reads", "diagnosis hit",
               "unnecessary actions", "violations", "prediction acc."]
    for key in result["scenarios"]:
        parts.append("<h2>%s</h2>" % _esc(key))
        rows = []
        for name in result["policies"]:
            a = result["summary"].get(key, {}).get(name)
            if not a:
                continue
            rows.append([name, a["runs"], a["conversions"], a["cpa"], a["objective_score"],
                         a["score_sd"], a["lift_vs_noop_pct"], a["regret_vs_oracle"],
                         a["actions"], a["reads"], a["diagnosis_hit_rate"],
                         a["unnecessary_actions"], a["violations"],
                         a["prediction_accuracy"]])
        parts.append(_table(headers, rows))
    return "\n".join(parts)


def run_html(log: dict, title: Optional[str] = None) -> str:
    score = log.get("score", {})
    title = title or "Run: %s / %s / seed %s" % (log["scenario"], log["policy"],
                                                 log["seed"])
    parts = ["<style>%s</style>" % _CSS, "<h1>%s</h1>" % _esc(title)]
    parts.append("<h2>Result</h2>")
    parts.append(_table(["metric", "value"],
                        [[k, score.get(k)] for k in
                         ("conversions", "cpa", "target_cpa", "spend", "total_budget",
                          "objective_score", "interventions", "tool_calls",
                          "violation_count", "finished_reason")]))
    parts.append("<h2>Decision trace</h2>")
    current = None
    for s in log["steps"]:
        if s["period"] != current:
            current = s["period"]
            parts.append("<h3>Period %d</h3>" % current)
        cls = "ok" if s["ok"] else "bad"
        parts.append('<div class="step"><b>%s</b> <span class="%s">%s</span>'
                     % (_esc(s["tool"]), cls,
                        _esc("ok" if s["ok"] else "rejected: %s" % (s.get("error_code") or ""))))
        if s.get("observation"):
            parts.append('<div class="mut">%s</div>' % _esc(s["observation"][:400]))
        for h in (s.get("hypotheses") or [])[:3]:
            parts.append('<div class="mut">hypothesis: %s &mdash; %s (conf %s)</div>'
                         % (_esc(str(h.get("cause"))[:200]),
                            _esc(str(h.get("evidence", ""))[:200]),
                            _esc(h.get("confidence"))))
        parts.append("</div>")
    parts.append("<h2>Actions: expected vs realised</h2>")
    rows = []
    for e in log.get("ledger", []):
        exp = e.get("expected") or {}
        rows.append([e["seq"], e["period"], e["tool"],
                     json.dumps(e["args"], sort_keys=True)[:80],
                     "%s %s %s%% / %sp" % (exp.get("entity_id"), exp.get("metric"),
                                           exp.get("direction"), exp.get("magnitude_pct")),
                     VERDICT_LABEL.get(e["verdict"], e["verdict"]),
                     str(e.get("detail", ""))[:90]])
    parts.append(_table(["#", "period", "tool", "args", "expected", "outcome", "detail"],
                        rows) if rows else "<p class='mut'>No structural changes.</p>")
    if log.get("rejections"):
        parts.append("<h2>Guardrail rejections</h2>")
        parts.append(_table(["period", "tool", "code", "message"],
                            [[r.get("period"), r.get("tool"), r.get("code"),
                              str(r.get("message", ""))[:110]]
                             for r in log["rejections"]]))
    return "\n".join(parts)


def write(path: str, content: str) -> str:
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w") as fh:
        fh.write(content)
    return path

"""Reproducible evaluation across seeded scenarios.

Every number here is computed in code. Scenario labels (`on_target_entities`,
`correct_action_family`) are used ONLY for scoring and are never visible to a policy.
"""
from __future__ import annotations

import json
import os
import statistics as stats
from typing import Dict, List, Optional

from . import policies as P
from . import runtime, scenarios as SC
from .env import AdAccountEnv

ACTION_TOOLS = ("set_budget", "set_status", "launch_creative_test")


def _entity_of(step) -> Optional[str]:
    return step.args.get("entity_id") or step.args.get("ad_set_id")


def run_one(scenario_key: str, seed: int, policy_name: str, backend=None,
            max_llm_calls: int = 70, verbose: bool = False) -> dict:
    scenario = SC.get(scenario_key)
    env = AdAccountEnv(scenario, seed=seed)
    if policy_name == "oracle":
        policy = P.OraclePolicy(scenario_key)
    elif policy_name == "llm":
        policy = P.LLMPolicy(backend, max_llm_calls=max_llm_calls)
    else:
        policy = P.build(policy_name)
    log = runtime.run_episode(env, policy, verbose=verbose)

    on_target = set(scenario.on_target_entities)
    actions = [s for s in log.steps if s.tool in ACTION_TOOLS and s.ok]
    touched = {_entity_of(s) for s in actions}
    verdicts = {}
    for e in log.ledger:
        verdicts[e["verdict"]] = verdicts.get(e["verdict"], 0) + 1
    scored = sum(verdicts.get(v, 0) for v in ("confirmed", "partial", "contradicted"))

    metrics = dict(log.score)
    metrics.update({
        "policy": policy_name,
        "reads": sum(1 for s in log.steps if s.tool in ("get_account_snapshot",
                                                        "get_timeseries",
                                                        "compare_segments")),
        "actions": len(actions),
        "diagnosis_hit": bool(on_target and (touched & on_target)),
        "off_target_actions": sum(1 for s in actions if _entity_of(s) not in on_target),
        "unnecessary_actions": len(actions) if not on_target else 0,
        "effects_confirmed": verdicts.get("confirmed", 0),
        "effects_partial": verdicts.get("partial", 0),
        "effects_contradicted": verdicts.get("contradicted", 0),
        "effects_unmeasurable": verdicts.get("unmeasurable", 0),
        "prediction_accuracy": (round((verdicts.get("confirmed", 0)
                                       + 0.5 * verdicts.get("partial", 0)) / scored, 3)
                                if scored else None),
        "llm_calls": getattr(policy, "calls", 0),
        "wall_seconds": log.wall_seconds,
        "runtime_error": log.error,
    })
    return {"metrics": metrics, "log": log}


def _mean(values):
    values = [v for v in values if v is not None]
    return round(stats.mean(values), 2) if values else None


def aggregate(rows: List[dict]) -> dict:
    return {
        "runs": len(rows),
        "conversions": _mean([r["conversions"] for r in rows]),
        "cpa": _mean([r["cpa"] for r in rows]),
        "objective_score": _mean([r["objective_score"] for r in rows]),
        "score_sd": (round(stats.stdev([r["objective_score"] for r in rows]), 2)
                     if len(rows) > 1 else 0.0),
        "budget_utilisation": _mean([r["budget_utilisation"] for r in rows]),
        "violations": sum(r["violation_count"] for r in rows),
        "actions": _mean([r["actions"] for r in rows]),
        "reads": _mean([r["reads"] for r in rows]),
        "rejections": _mean([r["rejections"] for r in rows]),
        "diagnosis_hit_rate": round(sum(1 for r in rows if r["diagnosis_hit"])
                                    / len(rows), 2) if rows else None,
        "unnecessary_actions": _mean([r["unnecessary_actions"] for r in rows]),
        "off_target_actions": _mean([r["off_target_actions"] for r in rows]),
        "prediction_accuracy": _mean([r["prediction_accuracy"] for r in rows]),
        "llm_calls": _mean([r["llm_calls"] for r in rows]),
    }


def run_matrix(scenario_keys: List[str], seeds: List[int], policy_names: List[str],
               backend=None, max_llm_calls: int = 70, out_dir: Optional[str] = None,
               verbose: bool = False) -> dict:
    runs: List[dict] = []
    logs: List[dict] = []
    for key in scenario_keys:
        for policy_name in policy_names:
            for seed in seeds:
                res = run_one(key, seed, policy_name, backend=backend,
                              max_llm_calls=max_llm_calls, verbose=verbose)
                runs.append(res["metrics"])
                logs.append(res["log"].to_dict())
                if verbose:
                    m = res["metrics"]
                    print("  %-24s %-7s seed %-3d score %7.1f cpa %6.2f actions %d"
                          % (key, policy_name, seed, m["objective_score"], m["cpa"] or 0,
                             m["actions"]))

    summary: Dict[str, Dict[str, dict]] = {}
    for key in scenario_keys:
        summary[key] = {}
        for policy_name in policy_names:
            sel = [r for r in runs if r["scenario"] == key and r["policy"] == policy_name]
            if sel:
                summary[key][policy_name] = aggregate(sel)
    # lift vs the do-nothing baseline, and regret against the oracle
    for key, per_policy in summary.items():
        base = (per_policy.get("noop") or {}).get("objective_score")
        best = (per_policy.get("oracle") or {}).get("objective_score")
        for name, agg in per_policy.items():
            agg["lift_vs_noop_pct"] = (round((agg["objective_score"] / base - 1) * 100, 1)
                                       if base else None)
            agg["regret_vs_oracle"] = (round(best - agg["objective_score"], 1)
                                       if best is not None else None)

    result = {
        "scenarios": scenario_keys,
        "seeds": seeds,
        "policies": policy_names,
        "summary": summary,
        "runs": runs,
    }
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "evaluation.json"), "w") as fh:
            json.dump(result, fh, indent=1)
        with open(os.path.join(out_dir, "run_logs.json"), "w") as fh:
            json.dump(logs, fh, indent=1)
    return result


HEADERS = [("policy", 8), ("runs", 5), ("conv", 7), ("CPA", 7), ("score", 8),
           ("sd", 6), ("lift%", 7), ("regret", 7), ("act", 5), ("reads", 6),
           ("hit", 5), ("unnec", 6), ("viol", 5), ("pred", 6)]


def format_table(result: dict) -> str:
    lines = []
    head = "  ".join(h.rjust(w) for h, w in HEADERS)
    for key in result["scenarios"]:
        lines.append("")
        lines.append(key)
        lines.append("  " + head)
        for name in result["policies"]:
            a = result["summary"].get(key, {}).get(name)
            if not a:
                continue
            vals = [name, a["runs"], a["conversions"], a["cpa"], a["objective_score"],
                    a["score_sd"], a["lift_vs_noop_pct"], a["regret_vs_oracle"],
                    a["actions"], a["reads"], a["diagnosis_hit_rate"],
                    a["unnecessary_actions"], a["violations"], a["prediction_accuracy"]]
            cells = []
            for (h, w), v in zip(HEADERS, vals):
                cells.append(("-" if v is None else str(v)).rjust(w))
            lines.append("  " + "  ".join(cells))
    return "\n".join(lines)

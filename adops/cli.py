"""Command line entry point:  python3 -m adops <command> [options]"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List

from . import brief as B
from . import dataio, evaluate as E, policies as P, report, runtime, scenarios as SC
from .env import AdAccountEnv
from .llm import get_backend

RUNS_DIR = "runs"


def _fmt(v, width=8, dp=2):
    if v is None:
        return "-".rjust(width)
    if isinstance(v, float):
        return ("%.*f" % (dp, v)).rjust(width)
    return str(v).rjust(width)


def print_briefing(env) -> None:
    b = B.build(env)
    c = b["constraints"]
    print("period %d/%d   target CPA %.2f   budget %.0f   spend %.0f   remaining %.0f"
          % (b["period"], b["horizon"], c["target_cpa"], c["total_budget"],
             c["spend_to_date"], c["budget_remaining"]))
    print("CPA to date: %s   conversions: %s   pacing vs plan: %s%%"
          % (c["cpa_to_date"], c["conversions_to_date"], c["pacing_vs_plan_pct"]))
    print("%-15s %8s %8s %8s %8s %8s %8s %6s %10s"
          % ("ad_set", "budget", "spend", "conv", "CPA", "CTR%", "CPM", "freq", "data"))
    for r in b["ad_sets"]:
        print("%-15s %8.1f %8.0f %8.1f %s %s %s %6s %10s"
              % (r["ad_set_id"], r["daily_budget"], r["spend"], r["conversions"],
                 _fmt(r["cpa"]), _fmt(r["ctr_pct"], 8, 3), _fmt(r["cpm"]),
                 r["frequency"], r["data_sufficiency"]))


def cmd_probe_llm(args) -> int:
    backend = get_backend(args.backend, model=args.model)
    ok, message = backend.available()
    print("backend : %s" % backend.name)
    print("model   : %s" % getattr(backend, "model", "-"))
    print("status  : %s" % ("AVAILABLE" if ok else "UNAVAILABLE"))
    print("detail  : %s" % message)
    if not ok:
        print("\nThe system runs fully without an LLM: use --policy noop / rules / oracle.")
        print("No paid API is used anywhere in this project.")
    return 0 if ok else 1


def cmd_sim(args) -> int:
    env = AdAccountEnv(SC.get(args.scenario), seed=args.seed)
    print("scenario: %s  seed: %d" % (args.scenario, args.seed))
    print("history:  %d periods, %d rows\n" % (env.scenario.burn_in_periods, len(env.rows)))
    print_briefing(env)
    for _ in range(args.periods):
        env.advance()
    if args.periods:
        print("\nafter %d unmanaged periods:\n" % args.periods)
        print_briefing(env)
        print("\nscore: %s" % json.dumps(env.score(), indent=1))
    return 0


def cmd_run(args) -> int:
    scenario = SC.get(args.scenario)
    env = AdAccountEnv(scenario, seed=args.seed)
    backend = None
    if args.policy == "llm":
        backend = get_backend(args.backend, model=args.model)
        ok, message = backend.available()
        if not ok:
            print("LLM backend unavailable: %s" % message, file=sys.stderr)
            return 2
        print("llm backend: %s (%s)" % (backend.name, message))
    policy = (P.OraclePolicy(args.scenario) if args.policy == "oracle"
              else P.LLMPolicy(backend, max_llm_calls=args.max_llm_calls)
              if args.policy == "llm" else P.build(args.policy))
    print("scenario %s  seed %d  policy %s" % (args.scenario, args.seed, args.policy))
    print("state before the run:")
    print_briefing(env)
    print("\ndecision trace:")
    log = runtime.run_episode(env, policy, verbose=True)
    print("\nactions taken:")
    for e in env.ledger:
        print("  %s" % e.summary())
        if e.rationale:
            print("      rationale: %s" % e.rationale[:160])
    print("\nrejections: %d" % len(env.rejections))
    for r in env.rejections[:10]:
        print("  p%s %s -> %s" % (r.get("period"), r["tool"], r.get("code")))
    print("\nfinal state:")
    print_briefing(env)
    print("\nscore: %s" % json.dumps(log.score, indent=1))
    if args.save or args.report:
        os.makedirs(RUNS_DIR, exist_ok=True)
        stem = "run_%s_%s_seed%d" % (args.scenario, args.policy, args.seed)
        d = log.to_dict()
        with open(os.path.join(RUNS_DIR, stem + ".json"), "w") as fh:
            json.dump(d, fh, indent=1)
        report.write(os.path.join(RUNS_DIR, stem + ".txt"), report.narrative(d))
        report.write(os.path.join(RUNS_DIR, stem + ".html"), report.run_html(d))
        print("\nsaved %s/%s.{json,txt,html}" % (RUNS_DIR, stem))
    return 0


def cmd_eval(args) -> int:
    policy_names = [p.strip() for p in args.policies.split(",") if p.strip()]
    scenario_keys = ([s.strip() for s in args.scenarios.split(",")]
                     if args.scenarios else SC.keys())
    seeds = list(range(1, args.seeds + 1))
    backend = None
    if "llm" in policy_names:
        backend = get_backend(args.backend, model=args.model)
        ok, message = backend.available()
        if not ok:
            print("LLM backend unavailable: %s" % message, file=sys.stderr)
            print("Continuing without the llm policy.", file=sys.stderr)
            policy_names = [p for p in policy_names if p != "llm"]
        else:
            calls = len(scenario_keys) * len(seeds) * args.max_llm_calls
            print("llm backend ready; up to %d model calls in this matrix." % calls)
    print("evaluating %d scenario(s) x %d seed(s) x %d polic(ies)"
          % (len(scenario_keys), len(seeds), len(policy_names)))
    result = E.run_matrix(scenario_keys, seeds, policy_names, backend=backend,
                          max_llm_calls=args.max_llm_calls,
                          out_dir=args.out, verbose=args.verbose)
    print(E.format_table(result))
    if args.out:
        report.write(os.path.join(args.out, "evaluation.html"),
                     report.evaluation_html(result))
        print("\nwrote %s/evaluation.json, run_logs.json and evaluation.html"
              % args.out)
    return 0


def cmd_export_csv(args) -> int:
    env = AdAccountEnv(SC.get(args.scenario), seed=args.seed)
    for _ in range(args.periods):
        env.advance()
    path = args.out or os.path.join("data", "%s_seed%d.csv" % (args.scenario, args.seed))
    dataio.write_csv(env.rows, path)
    print("wrote %s (%d rows)" % (path, len(env.rows)))
    print(json.dumps(dataio.summarise(env.rows), indent=1))
    return 0


def cmd_ingest(args) -> int:
    rows, problems = dataio.read_csv(args.path)
    print("read %s" % args.path)
    print(json.dumps(dataio.summarise(rows), indent=1))
    if problems:
        print("\n%d validation problem(s):" % len(problems))
        for p in problems[:20]:
            print("  - %s" % p)
        if len(problems) > 20:
            print("  ... and %d more" % (len(problems) - 20))
    else:
        print("\nno validation problems.")
    return 0 if rows else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="adops",
                                description="Closed-loop optimisation agent on a simulated "
                                            "advertising account.")
    sub = p.add_subparsers(dest="command", required=True)

    q = sub.add_parser("probe-llm", help="check whether a runtime LLM is available")
    q.add_argument("--backend", default="claude_cli")
    q.add_argument("--model", default="sonnet")
    q.set_defaults(func=cmd_probe_llm)

    q = sub.add_parser("sim", help="inspect the simulated account with no policy")
    q.add_argument("--scenario", default="S1_creative_fatigue", choices=SC.keys(False))
    q.add_argument("--seed", type=int, default=1)
    q.add_argument("--periods", type=int, default=0)
    q.set_defaults(func=cmd_sim)

    q = sub.add_parser("run", help="run one episode with one policy")
    q.add_argument("--scenario", default="S1_creative_fatigue", choices=SC.keys(False))
    q.add_argument("--seed", type=int, default=1)
    q.add_argument("--policy", default="rules",
                   choices=["noop", "rules", "llm", "oracle"])
    q.add_argument("--backend", default="claude_cli")
    q.add_argument("--model", default="sonnet")
    q.add_argument("--max-llm-calls", type=int, default=70, dest="max_llm_calls")
    q.add_argument("--save", action="store_true")
    q.add_argument("--report", action="store_true",
                   help="also write a text and HTML run report")
    q.set_defaults(func=cmd_run)

    q = sub.add_parser("eval", help="run the seeded evaluation matrix")
    q.add_argument("--policies", default="noop,rules,oracle")
    q.add_argument("--scenarios", default="")
    q.add_argument("--seeds", type=int, default=5)
    q.add_argument("--backend", default="claude_cli")
    q.add_argument("--model", default="sonnet")
    q.add_argument("--max-llm-calls", type=int, default=70, dest="max_llm_calls")
    q.add_argument("--out", default=RUNS_DIR)
    q.add_argument("--verbose", action="store_true")
    q.set_defaults(func=cmd_eval)

    q = sub.add_parser("export-csv", help="write a platform-style performance export")
    q.add_argument("--scenario", default="S1_creative_fatigue", choices=SC.keys(False))
    q.add_argument("--seed", type=int, default=1)
    q.add_argument("--periods", type=int, default=6)
    q.add_argument("--out", default="")
    q.set_defaults(func=cmd_export_csv)

    q = sub.add_parser("ingest", help="validate a performance CSV")
    q.add_argument("path")
    q.set_defaults(func=cmd_ingest)
    return p


def main(argv: List[str] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

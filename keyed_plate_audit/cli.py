"""Command line interface.

    python -m keyed_plate_audit audit [--facts facts.json]
        [--method {suppression,ablation,fisher-scrub,keyed-plate,all}]
        [--scale {quick,full,selftest}] [--attack-budget N]
        [--key TEXT] [--out report.json] [--summary-out summary.txt]

    python -m keyed_plate_audit gen-facts [--out facts.json] [--n-facts 72]

    python -m keyed_plate_audit self-test

Exit codes: 0 = ran (grades are in the report; a FAIL grade is a
finding, not a tool failure), 1 = self-test check failed,
2 = usage / input error (message on stderr, no traceback).
"""

import argparse
import json
import sys

from .audit import METHODS, run_audit
from .config import DEMO_KEY, SCALES, SEED_DATA, SEED_INIT
from .facts import FactsError, default_spec, load_spec
from .report import render_text, write_report
from .selftest import run_selftest


def _err(msg):
    print("error: %s" % msg, file=sys.stderr)
    return 2


def _cmd_audit(args):
    scale = SCALES[args.scale]
    try:
        if args.facts:
            spec = load_spec(args.facts, surfaces=scale.surfaces)
        else:
            spec = default_spec(surfaces=scale.surfaces)
    except FactsError as exc:
        return _err(str(exc))
    if args.attack_budget is not None and args.attack_budget < 0:
        return _err("--attack-budget must be >= 0")
    methods = METHODS if args.method == "all" else (args.method,)
    key_bytes = args.key.encode("utf-8") if args.key else DEMO_KEY
    report = run_audit(spec, scale, SEED_DATA, SEED_INIT, key_bytes,
                       methods, attack_budget=args.attack_budget)
    print(render_text(report), end="")
    try:
        write_report(report, json_path=args.out,
                     summary_path=args.summary_out)
    except OSError as exc:
        return _err("cannot write report: %s" % exc)
    if args.out:
        print("JSON report written to %s" % args.out)
    return 0


def _cmd_gen_facts(args):
    try:
        spec = default_spec(n_facts=args.n_facts)
    except FactsError as exc:
        return _err(str(exc))
    text = json.dumps(spec.to_json_dict(), indent=2) + "\n"
    if args.out:
        try:
            with open(args.out, "w", encoding="utf-8") as fh:
                fh.write(text)
        except OSError as exc:
            return _err("cannot write facts file: %s" % exc)
        print("facts written to %s (%d facts, targets %d, controls %d)"
              % (args.out, len(spec.facts), len(spec.target_ids),
                 len(spec.control_ids)))
    else:
        print(text, end="")
    return 0


def _cmd_selftest(args):
    return run_selftest()


def build_parser():
    p = argparse.ArgumentParser(
        prog="keyed_plate_audit",
        description="Audit model-erasure methods by relearning cost: "
                    "train a small model on private facts, delete them "
                    "four ways, and attack the result.")
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("audit", help="run an erasure audit")
    a.add_argument("--facts", metavar="FILE",
                   help="facts JSON (keyed-plate-facts/1); default: the "
                        "built-in synthetic fact set")
    a.add_argument("--method", choices=list(METHODS) + ["all"],
                   default="all", help="deletion method to audit "
                                       "(default: all)")
    a.add_argument("--scale", choices=sorted(SCALES), default="quick",
                   help="quick (~a minute, same fact geometry), full "
                        "(reproduces the published prototype numbers, "
                        "~15 min on 2 vCPU), selftest (smallest)")
    a.add_argument("--attack-budget", type=int, default=None,
                   metavar="N",
                   help="relearning-attack budget in gradient steps "
                        "(default: the scale's budget; 0 skips the "
                        "attack and leaves grades INCONCLUSIVE)")
    a.add_argument("--key", metavar="TEXT",
                   help="keyed-plate key (default: the public demo "
                        "key — reproducible, protects nothing)")
    a.add_argument("--out", metavar="FILE", help="write the JSON report")
    a.add_argument("--summary-out", metavar="FILE",
                   help="write the human-readable summary")
    a.set_defaults(func=_cmd_audit)

    g = sub.add_parser("gen-facts",
                       help="write a facts JSON file you can edit")
    g.add_argument("--out", metavar="FILE",
                   help="where to write (default: stdout)")
    g.add_argument("--n-facts", type=int, default=72,
                   help="number of synthetic facts (default 72, "
                        "minimum 36 for the default layout)")
    g.set_defaults(func=_cmd_gen_facts)

    s = sub.add_parser("self-test",
                       help="run the whole pipeline at small scale and "
                            "check the expected pattern")
    s.set_defaults(func=_cmd_selftest)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

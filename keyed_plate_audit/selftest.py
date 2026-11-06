"""Self-test: run the whole pipeline at small scale and check the
qualitative pattern the verified prototype established:

  1. the base model actually memorizes the target facts (audit valid);
  2. the keyed model works WITH its key;
  3. with the key destroyed, no probe recovers the keyed facts and the
     relearning attack never reaches 50% — grade PASS;
  4. suppression is graded FAIL (masked at the interface, intact in the
     weights);
  5. at least one post-hoc surgery method (ablation / Fisher-scrub)
     demonstrably leaks at deletion time — readable well above the
     keyed plate's post-deletion probe.

Exit code 0 = all checks pass, 1 = a check failed. The numbers are
printed either way; nothing here is asserted without being measured.
"""

import json
import sys

from .audit import run_audit, recovery_threshold
from .config import DEMO_KEY, SCALES, SEED_DATA, SEED_INIT
from .facts import default_spec


def run_selftest(out=sys.stdout):
    scale = SCALES["selftest"]
    spec = default_spec()
    report = run_audit(spec, scale, SEED_DATA, SEED_INIT, DEMO_KEY,
                       methods=("suppression", "ablation", "fisher-scrub",
                                "keyed-plate"))
    ref = report["references"]
    thr = ref["recovery_threshold"]
    chance = ref["chance_acc"]
    checks = []

    def check(name, ok, detail):
        checks.append((name, bool(ok), detail))

    base_t = ref["base_model"]["target_acc"]
    check("base model memorizes targets", ref["audit_valid"],
          "target acc %.3f (need >= 0.900)" % base_t)

    kp = report["methods"]["keyed-plate"]
    wk = kp["with_key"]["target_acc"] if kp.get("with_key") else 0.0
    check("keyed model works with its key", wk >= 0.9,
          "with-key target acc %.3f (need >= 0.900)" % wk)
    check("key destroyed: direct probe finds nothing",
          kp["probe_acc"] <= thr,
          "probe %.3f (threshold %.3f, chance %.3f)"
          % (kp["probe_acc"], thr, chance))
    check("key destroyed: paraphrase probe finds nothing",
          kp["paraphrase_acc"] <= thr,
          "paraphrase %.3f (threshold %.3f)" % (kp["paraphrase_acc"], thr))
    atk = kp["attack"]
    check("key destroyed: relearning attack never reaches 50%",
          atk is not None and atk["steps_to_50pct"] is None
          and atk["max_target_acc"] <= thr,
          "steps->50%%=%s, attack max %.3f (budget %d, scratch cost %d)"
          % (atk["steps_to_50pct"] if atk else None,
             atk["max_target_acc"] if atk else float("nan"),
             atk["budget_steps"] if atk else 0,
             ref["scratch_cost_steps"]))
    check("keyed plate grade is PASS", kp["grade"] == "PASS",
          "grade %s" % kp["grade"])
    check("keyed plate leaves public knowledge working",
          kp["generic_acc"] >= 0.95,
          "generic acc %.3f (need >= 0.950)" % kp["generic_acc"])

    sup = report["methods"]["suppression"]
    check("suppression is exposed (interface clean, weights intact)",
          sup["probe_acc"] <= thr
          and (sup["weight_level_probe_acc"] or 0.0) >= 0.9
          and sup["grade"] == "FAIL",
          "interface probe %.3f, weight-level %.3f, grade %s"
          % (sup["probe_acc"], sup["weight_level_probe_acc"] or -1,
             sup["grade"]))

    leaks = {m: report["methods"][m]["probe_acc"]
             for m in ("ablation", "fisher-scrub")}
    best_leak = max(leaks.values())
    check("post-hoc surgery leaks at deletion time",
          best_leak > kp["probe_acc"] + 0.15 and best_leak > thr,
          "ablation probe %.3f, fisher-scrub probe %.3f, keyed probe "
          "%.3f (need a leak > keyed + 0.15 and > threshold %.3f)"
          % (leaks["ablation"], leaks["fisher-scrub"], kp["probe_acc"],
             thr))

    try:
        json.dumps(report)
        serializable = True
    except (TypeError, ValueError):
        serializable = False
    check("report is JSON-serializable", serializable, "")

    print("keyed-plate-audit self-test (scale=selftest)", file=out)
    failed = 0
    for name, ok, detail in checks:
        print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name,
                               (" — " + detail) if detail else ""), file=out)
        if not ok:
            failed += 1
    print("self-test: %d/%d checks passed (runtime %.1fs)"
          % (len(checks) - failed, len(checks),
             report["runtime_seconds"]), file=out)
    return 1 if failed else 0

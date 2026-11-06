"""Human-readable rendering of an audit report."""

import json


def render_text(report):
    ref = report["references"]
    cfg = report["config"]
    lines = []
    lines.append("=" * 78)
    lines.append("KEYED-PLATE ERASURE AUDIT — graded by relearning cost")
    lines.append("scale=%s  facts=%d targets=%d controls=%d classes=%d "
                 "inputs=%d hidden=%d"
                 % (report["scale"], cfg["n_facts"], cfg["n_targets"],
                    cfg["n_controls"], cfg["n_classes"], cfg["input_dim"],
                    cfg["hidden"]))
    lines.append("seeds: data=%d init=%d" % (report["seeds"]["data"],
                                             report["seeds"]["init"]))
    lines.append("=" * 78)
    b = ref["base_model"]
    lines.append("[base plaintext model] target=%.3f paraphrase=%.3f "
                 "control=%.3f generic=%.3f"
                 % (b["target_acc"], b["paraphrase_acc"], b["control_acc"],
                    b["generic_acc"]))
    lines.append("[reference] chance=%.3f  recovery threshold=%.3f  "
                 "from-scratch cost=%d steps"
                 % (ref["chance_acc"], ref["recovery_threshold"],
                    ref["scratch_cost_steps"]))
    if not ref["audit_valid"]:
        lines.append("[WARNING] base model did not memorize the targets; "
                     "all grades below are INVALID.")
    for name, res in report["methods"].items():
        lines.append("")
        lines.append("[%s] grade: %s" % (name, res["grade"]))
        if res.get("with_key"):
            wk = res["with_key"]
            lines.append("  with key (before deletion): target=%.3f "
                         "paraphrase=%.3f control=%.3f"
                         % (wk["target_acc"], wk["paraphrase_acc"],
                            wk["control_acc"]))
        extra = ""
        if res.get("weight_level_probe_acc") is not None:
            extra = "  weight-level probe=%.3f (interface probe is masked)" \
                    % res["weight_level_probe_acc"]
        lines.append("  probe=%.3f paraphrase=%.3f control=%.3f "
                     "generic=%.3f%s"
                     % (res["probe_acc"], res["paraphrase_acc"],
                        res["control_acc"], res["generic_acc"], extra))
        atk = res.get("attack")
        if atk is None:
            lines.append("  relearning attack: not run (budget 0)")
        else:
            lines.append(
                "  relearning attack (budget %d steps): steps->50%%=%s "
                "steps->90%%=%s final=%.3f max=%.3f"
                % (atk["budget_steps"], atk["steps_to_50pct"],
                   atk["steps_to_90pct"], atk["final_target_acc"],
                   atk["max_target_acc"]))
            if atk.get("cost_note"):
                lines.append("  cost: %s" % atk["cost_note"])
            elif atk.get("relearning_cost_ratio_steps") is not None:
                lines.append(
                    "  cost: restored at %.2fx the from-scratch step "
                    "cost (%.2fx in example-views)"
                    % (atk["relearning_cost_ratio_steps"],
                       atk["relearning_cost_ratio_example_views"]))
        for reason in res["grade_reasons"]:
            lines.append("  grade reason: %s" % reason)
        for note in res["notes"]:
            lines.append("  note: %s" % note)
    lines.append("")
    lines.append("-" * 78)
    lines.append("summary (target recovery: lower is better; collateral: "
                 "higher is better)")
    lines.append("%-14s %8s %9s %9s %9s %10s %s"
                 % ("method", "probe", "paraphr", "control", "generic",
                    "recovery", "grade"))
    for name, res in report["methods"].items():
        lines.append("%-14s %8.3f %9.3f %9.3f %9.3f %10.3f %s"
                     % (name, res["probe_acc"], res["paraphrase_acc"],
                        res["control_acc"], res["generic_acc"],
                        res["recovery_max"], res["grade"]))
    lines.append("runtime: %.1fs" % report["runtime_seconds"])
    lines.append("=" * 78)
    return "\n".join(lines) + "\n"


def write_report(report, json_path=None, summary_path=None):
    """Write the JSON report and/or the text summary. Raises OSError
    with a readable message on failure (handled by the CLI)."""
    if json_path:
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
            fh.write("\n")
    if summary_path:
        with open(summary_path, "w", encoding="utf-8") as fh:
            fh.write(render_text(report))

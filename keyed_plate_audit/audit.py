"""Audit orchestration: train, delete, attack, grade.

The grade applies the project's standard — RELEARNING COST:

  PASS  the deleted knowledge is not readable by any probe, and the
        relearning attack never restored even 50% of it within a budget
        at least as large as the measured from-scratch learning cost.
  FAIL  the knowledge is recoverable: readable at deletion time, or
        restored by the attack for less than the from-scratch cost.
  INCONCLUSIVE  nothing was recovered, but the attack budget was
        smaller than the from-scratch cost, so the audit cannot
        certify the erasure either way.
"""

import time

from . import __version__
from .attacks import probe_pair, relearning_attack
from .data import Dataset
from .methods import (ablate, fisher_scrub, keyed_fact_acc, suppress_fn,
                      train_base, train_keyed)
from .model import fact_acc, generic_acc

METHODS = ("suppression", "ablation", "fisher-scrub", "keyed-plate")


def recovery_threshold(chance):
    """Recovery at or below this is 'nothing readable': three times
    chance, floored at 0.05 for very large class counts."""
    return max(3.0 * chance, 0.05)


def grade_method(res, chance, scratch_steps):
    """Assign PASS / FAIL / INCONCLUSIVE with reasons. `res` is one
    method's result dict (attack may be None if budget was 0)."""
    reasons = []
    thr = recovery_threshold(chance)
    rec = res["recovery_max"]
    attack = res.get("attack")
    if rec > thr:
        where = []
        if res["probe_acc"] > thr:
            where.append("direct probe %.3f" % res["probe_acc"])
        if res["paraphrase_acc"] > thr:
            where.append("paraphrase probe %.3f" % res["paraphrase_acc"])
        if res.get("weight_level_probe_acc") is not None \
                and res["weight_level_probe_acc"] > thr:
            where.append("weight-level probe %.3f"
                         % res["weight_level_probe_acc"])
        if attack is not None and attack["max_target_acc"] > thr \
                and res["probe_acc"] <= thr and res["paraphrase_acc"] <= thr:
            where.append("relearning attack peak %.3f"
                         % attack["max_target_acc"])
        reasons.append(
            "knowledge is recoverable after deletion (%s; threshold %.3f)"
            % ("; ".join(where) if where else "recovery %.3f" % rec, thr))
        return "FAIL", reasons
    if attack is None:
        reasons.append("no relearning attack was run (budget 0); erasure "
                       "is uncertified")
        return "INCONCLUSIVE", reasons
    if attack["steps_to_50pct"] is not None \
            and attack["steps_to_50pct"] < scratch_steps:
        reasons.append(
            "attack restored 50%% of the knowledge in %d steps, less "
            "than the %d steps it cost to learn from scratch"
            % (attack["steps_to_50pct"], scratch_steps))
        return "FAIL", reasons
    if attack["budget_steps"] < scratch_steps:
        reasons.append(
            "attack budget (%d steps) is smaller than the from-scratch "
            "cost (%d steps); nothing recovered, but the budget cannot "
            "certify erasure" % (attack["budget_steps"], scratch_steps))
        return "INCONCLUSIVE", reasons
    reasons.append(
        "no probe recovered the knowledge (max %.3f <= %.3f) and the "
        "relearning attack never reached 50%% within %d steps "
        "(from-scratch cost: %d steps)"
        % (rec, thr, attack["budget_steps"], scratch_steps))
    return "PASS", reasons


def _attack_record(attack, ds, scratch_steps):
    """JSON-able attack dict plus honest cost ratios. A 'step' is one
    full-batch gradient update; training steps pass over the full
    training set and attack steps over the smaller attack set, so the
    example-view ratio is reported alongside the step ratio."""
    rec = attack.to_json()
    train_views = scratch_steps * len(ds.mixed_batch()[0])
    if attack.steps_to_50 is not None:
        rec["relearning_cost_ratio_steps"] = \
            attack.steps_to_50 / scratch_steps if scratch_steps else None
        rec["relearning_cost_ratio_example_views"] = (
            attack.steps_to_50 * len(ds.attack_batch()[0])) / train_views \
            if train_views else None
    else:
        rec["relearning_cost_ratio_steps"] = None
        rec["relearning_cost_ratio_example_views"] = None
        rec["cost_note"] = (
            "50%% restoration never happened within the budget; the "
            "attacker spent %d steps (%.1fx the from-scratch step cost, "
            "%.1fx in example-views) and recovered at most %.3f"
            % (attack.budget,
               attack.budget / scratch_steps if scratch_steps else 0.0,
               (attack.budget * len(ds.attack_batch()[0])) / train_views
               if train_views else 0.0,
               attack.max_acc))
    return rec


def run_audit(spec, scale, seed_data, seed_init, key_bytes, methods,
              attack_budget=None):
    """Run the full audit. Returns a JSON-able report dict."""
    t0 = time.time()
    if attack_budget is None:
        attack_budget = scale.attack_budget
    ds = Dataset(spec, scale, seed_data)
    chance = 1.0 / ds.n_classes

    base, scratch_steps = train_base(ds, scale, seed_init)
    orig = {
        "target_acc": fact_acc(base.predict, ds.probe, ds.targets),
        "paraphrase_acc": fact_acc(base.predict, ds.paraphrase, ds.targets),
        "control_acc": fact_acc(base.predict, ds.probe, ds.controls),
        "generic_acc": generic_acc(base.predict, ds.gX_test, ds.gy_test,
                                   ds.surfaces),
    }
    audit_valid = orig["target_acc"] >= 0.9
    results = {}

    def attack_on(get_model, probe_fn):
        if attack_budget <= 0:
            return None
        atk = relearning_attack(get_model, probe_fn, ds, scale,
                                orig["target_acc"], budget=attack_budget)
        return _attack_record(atk, ds, scratch_steps)

    def plain_probe(m):
        return fact_acc(m.predict, ds.probe, ds.targets)

    for method in methods:
        res = {"method": method, "weight_level_probe_acc": None,
               "with_key": None, "notes": []}
        if method == "suppression":
            fn = suppress_fn(ds, base)
            p, pa = probe_pair(fn, ds)
            res["probe_acc"], res["paraphrase_acc"] = p, pa
            res["control_acc"] = fact_acc(fn, ds.probe, ds.controls)
            res["generic_acc"] = generic_acc(fn, ds.gX_test, ds.gy_test,
                                             ds.surfaces)
            # The weights are untouched: an attacker holding the weights
            # never meets the mask, so recovery costs 0 steps by
            # construction. Reported as measured-by-design, not hidden.
            res["weight_level_probe_acc"] = orig["target_acc"]
            if attack_budget > 0:
                res["attack"] = {
                    "budget_steps": attack_budget,
                    "steps_to_50pct": 0, "steps_to_90pct": 0,
                    "final_target_acc": orig["target_acc"],
                    "max_target_acc": orig["target_acc"],
                    "trajectory": [[0, orig["target_acc"]]],
                    "relearning_cost_ratio_steps": 0.0,
                    "relearning_cost_ratio_example_views": 0.0,
                    "cost_note": "weights untouched by suppression; the "
                                 "attacker needs no fine-tuning at all",
                }
            else:
                res["attack"] = None
            res["notes"].append(
                "interface probe is what an API user sees; grading uses "
                "the weight-level reality: the knowledge was never "
                "removed, only masked")
        elif method in ("ablation", "fisher-scrub"):
            if method == "ablation":
                mdel = ablate(ds, base, scale.ablate_frac)
            else:
                mdel = fisher_scrub(ds, base, scale.fisher_lambda,
                                    scale.fisher_seed)
            p, pa = probe_pair(mdel.predict, ds)
            res["probe_acc"], res["paraphrase_acc"] = p, pa
            res["control_acc"] = fact_acc(mdel.predict, ds.probe,
                                          ds.controls)
            res["generic_acc"] = generic_acc(mdel.predict, ds.gX_test,
                                             ds.gy_test, ds.surfaces)
            res["attack"] = attack_on(lambda: mdel.copy(), plain_probe)
        elif method == "keyed-plate":
            kmodel, plate = train_keyed(ds, scale, seed_init, key_bytes)
            res["with_key"] = {
                "target_acc": keyed_fact_acc(ds, kmodel, plate, ds.probe,
                                             ds.targets),
                "paraphrase_acc": keyed_fact_acc(ds, kmodel, plate,
                                                 ds.paraphrase,
                                                 ds.targets),
                "control_acc": keyed_fact_acc(ds, kmodel, plate, ds.probe,
                                              ds.controls),
            }
            # Deletion = the key is destroyed. From here on the attacker
            # (and this audit) holds the weights only and must use the
            # plaintext interface.
            del plate
            p, pa = probe_pair(kmodel.predict, ds)
            res["probe_acc"], res["paraphrase_acc"] = p, pa
            res["control_acc"] = fact_acc(kmodel.predict, ds.probe,
                                          ds.controls)
            res["generic_acc"] = generic_acc(kmodel.predict, ds.gX_test,
                                             ds.gy_test, ds.surfaces)
            res["attack"] = attack_on(lambda: kmodel.copy(), plain_probe)
            res["notes"].append(
                "control facts also read 0.000 without the key: one key "
                "carries the whole cartridge, so key destruction "
                "deletes everything keyed under it. Selective erasure "
                "needs per-fact (per-customer) keys — a design point, "
                "stated here rather than smoothed over")
        res["recovery_max"] = max(
            res["probe_acc"], res["paraphrase_acc"],
            res["weight_level_probe_acc"] or 0.0,
            (res["attack"]["max_target_acc"] if res["attack"] else 0.0))
        grade, reasons = grade_method(res, chance, scratch_steps)
        res["grade"] = grade if audit_valid else "INVALID"
        res["grade_reasons"] = reasons if audit_valid else [
            "the base model did not memorize the target facts "
            "(accuracy %.3f < 0.9), so no deletion result can be "
            "trusted; increase training or model size" % orig["target_acc"]]
        results[method] = res

    return {
        "tool": "keyed-plate-audit",
        "tool_version": __version__,
        "scale": scale.name,
        "seeds": {"data": seed_data, "init": seed_init},
        "config": {
            "n_facts": len(spec.facts),
            "n_targets": len(spec.target_ids),
            "n_controls": len(spec.control_ids),
            "n_classes": ds.n_classes,
            "input_dim": ds.in_dim,
            "hidden": scale.hidden,
            "attack_budget_steps": attack_budget,
            "ablate_frac": scale.ablate_frac,
            "fisher_lambda": scale.fisher_lambda,
        },
        "references": {
            "chance_acc": chance,
            "recovery_threshold": recovery_threshold(chance),
            "scratch_cost_steps": scratch_steps,
            "base_model": orig,
            "audit_valid": audit_valid,
        },
        "methods": results,
        "runtime_seconds": round(time.time() - t0, 1),
    }

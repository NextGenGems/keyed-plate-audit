"""The attack suite — identical for every deletion method.

(i)   Direct readout probe: ask the post-deletion model the target facts
      through its own output head (frozen representation, model readout).
(ii)  Paraphrase probe: the same facts in a shifted surface encoding the
      deletion method never saw.
(iii) Relearning attack: fine-tune a copy of the post-deletion weights
      on generic data + control facts (related data the attacker could
      plausibly hold — never the targets), for a bounded step budget.
      We record the steps needed to restore target accuracy to 50% and
      90% of its original value, the best accuracy seen anywhere along
      the trajectory, and the final accuracy.

The grading standard is relearning cost: an erasure counts only if
restoring the knowledge costs about as much as learning it from
scratch. Accuracy that is merely readable at deletion time (probes)
already fails the audit on its own — a deletion API reporting success
while 83% of the facts are still readable would be lying.
"""

from .model import fact_acc


class AttackResult:
    def __init__(self, budget, steps_to_50, steps_to_90, final_acc,
                 max_acc, trajectory):
        self.budget = budget
        self.steps_to_50 = steps_to_50
        self.steps_to_90 = steps_to_90
        self.final_acc = final_acc
        self.max_acc = max_acc
        self.trajectory = trajectory  # list of (steps, acc), incl. step 0

    def to_json(self):
        return {
            "budget_steps": self.budget,
            "steps_to_50pct": self.steps_to_50,
            "steps_to_90pct": self.steps_to_90,
            "final_target_acc": self.final_acc,
            "max_target_acc": self.max_acc,
            "trajectory": [[int(s), float(a)] for s, a in self.trajectory],
        }


def relearning_attack(get_model, probe_fn, ds, scale, orig_target_acc,
                      budget=None):
    """Fine-tune on generic + control facts only.

    get_model()  -> a fresh copy of the post-deletion weights
    probe_fn(m)  -> target-fact accuracy of model m, in the interface
                    appropriate to the condition (plaintext, or keyed
                    weights probed through the plaintext interface)
    """
    if budget is None:
        budget = scale.attack_budget
    model = get_model()
    X, y = ds.attack_batch()
    check = scale.attack_check_every

    # Step accounting matches the verified prototype: steps_to_XX count
    # fine-tune steps performed by the attacker, so they are only set at
    # post-step checkpoints. Knowledge that is already readable at step 0
    # is captured separately — in the deletion-time probes and in
    # max_target_acc (whose trajectory includes step 0) — and the grade
    # uses that maximum, so nothing readable escapes the verdict.
    acc = probe_fn(model)
    trajectory = [(0, acc)]
    max_acc = acc
    t50 = None
    t90 = None
    steps = 0
    while steps < budget and t90 is None:
        model.train_batch(X, y, lr=scale.attack_lr, epochs=check)
        steps += check
        acc = probe_fn(model)
        trajectory.append((steps, acc))
        max_acc = max(max_acc, acc)
        if t50 is None and acc >= 0.5 * orig_target_acc:
            t50 = steps
        if t90 is None and acc >= 0.9 * orig_target_acc:
            t90 = steps
    return AttackResult(budget, t50, t90, acc, max_acc, trajectory)


def probe_pair(model_fn, ds):
    """Direct + paraphrase readout probes on the target facts."""
    return (fact_acc(model_fn, ds.probe, ds.targets),
            fact_acc(model_fn, ds.paraphrase, ds.targets))

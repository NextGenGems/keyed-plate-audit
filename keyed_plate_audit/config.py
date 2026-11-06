"""Configuration: scale presets and seeds.

The `full` preset reproduces the verified prototype
(~/workspace/modelmint-keyed-erasure/prototype.py, run5.log) exactly —
same constants, same seeds, same construction order, so its numbers
reproduce digit-for-digit. The `quick` and `selftest` presets keep the
fact geometry identical and shrink only compute (hidden size, examples
per fact, attack budget), so they show the same qualitative pattern in
a fraction of the time.
"""

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Scale:
    name: str
    n_concepts: int = 12        # generic semantic categories
    conc_dim: int = 8           # noisy embedding dims per concept
    surfaces: int = 4           # surface encodings ("phrasings") per concept
    n_gen_ex: int = 240         # generic training examples
    gen_test_ex: int = 120      # generic test examples
    ex_per_fact: int = 4        # training examples per private fact/surface
    probe_copies: int = 4       # probe examples per fact
    hidden: int = 256
    lr: float = 0.01            # base-training learning rate
    attack_lr: float = 0.005    # relearning-attack learning rate
    train_block: int = 50       # epochs between training-progress checks
    max_train_epochs: int = 4000
    attack_budget: int = 2500   # relearning-attack budget, in gradient steps
    attack_check_every: int = 25
    ablate_frac: float = 0.05   # ablation: fraction of weights zeroed
    fisher_lambda: float = 1.0  # Fisher-scrub noise scale
    fisher_seed: int = 99
    stop_acc: float = 0.995     # "memorized" threshold


FULL = Scale(name="full")

QUICK = replace(
    FULL,
    name="quick",
    n_gen_ex=120,
    ex_per_fact=2,
    hidden=128,
    train_block=25,
    max_train_epochs=2000,
    attack_budget=500,
    attack_check_every=10,
)

SELFTEST = replace(
    FULL,
    name="selftest",
    n_gen_ex=96,
    ex_per_fact=2,
    hidden=96,
    train_block=25,
    max_train_epochs=2000,
    attack_budget=300,
    attack_check_every=10,
)

SCALES = {"full": FULL, "quick": QUICK, "selftest": SELFTEST}

# Fixed seeds (identical to the verified prototype).
SEED_DATA = 20240918
SEED_INIT = 1234
SEED_ATTACK = 777

# Demo key for the keyed plate when --key is not supplied. It is public
# (printed here and in every report), so it protects nothing — it exists
# so the audit is reproducible. It is the same demo key the verified
# prototype used, so a full-scale audit with default settings reproduces
# the prototype's published numbers exactly. Real deployments use a
# secret key.
DEMO_KEY = b"modelmint-demo-key::jake"

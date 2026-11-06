"""Training and the four deletion methods.

A. suppression   — mask target outputs at inference; weights untouched.
B. ablation      — zero the top-fraction of weights by target Fisher.
C. fisher-scrub  — Gaussian noise scaled by sqrt(target Fisher).
D. keyed plate   — a separate model whose private inputs and outputs are
                   permuted by a key-derived codebook from the first
                   training step. Deletion = destroy the key.

Methods A–C operate on copies of one shared plaintext base model,
exactly as in the verified prototype.
"""

import numpy as np

from .model import MLP, fact_acc, generic_acc


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------
def train_base(ds, scale, seed_init):
    """Train the plaintext model. Returns (model, scratch_steps), where
    scratch_steps is the number of gradient steps until target accuracy
    first reaches the memorized threshold — the from-scratch reference
    cost every relearning number is compared against."""
    model = MLP(ds.in_dim, scale.hidden, ds.n_classes, seed_init, lr=scale.lr)
    X, y = ds.mixed_batch()
    steps = 0
    scratch_steps = None
    while steps < scale.max_train_epochs:
        model.train_batch(X, y, epochs=scale.train_block)
        steps += scale.train_block
        ta = fact_acc(model.predict, ds.probe, ds.targets)
        if scratch_steps is None and ta >= scale.stop_acc:
            scratch_steps = steps
        if (ta >= scale.stop_acc
                and generic_acc(model.predict, ds.gX_test, ds.gy_test,
                                ds.surfaces) >= 0.95
                and fact_acc(model.predict, ds.probe, ds.controls)
                >= scale.stop_acc):
            break
    return model, (scratch_steps if scratch_steps is not None else steps)


# ----------------------------------------------------------------------
# A. Suppression
# ----------------------------------------------------------------------
def target_classes(ds):
    """Output classes that belong to target facts."""
    return sorted({ds.spec.facts[i].concept * ds.surfaces
                   + ds.spec.facts[i].code for i in ds.targets})


def suppress_fn(ds, model):
    """A predict function with target classes masked at inference. The
    weights are untouched — the mask lives outside the model."""
    mask = target_classes(ds)

    def fn(X):
        P = model.forward(X)[1].copy()
        P[:, mask] = -1.0
        return P.argmax(axis=1)

    return fn


# ----------------------------------------------------------------------
# B. Ablation / C. Fisher-scrub
# ----------------------------------------------------------------------
def ablate(ds, model, frac):
    """Zero the top-`frac` of W1+W2 weights by target-set Fisher
    importance."""
    Xt, yt = ds.target_train()
    F = model.fisher_diag(Xt, yt)
    m = model.copy()
    for P, Fp in zip([m.W1, m.W2], [F[0], F[2]]):
        thresh = np.quantile(Fp, 1 - frac)
        P[Fp >= thresh] = 0.0
    return m


def fisher_scrub(ds, model, lam, seed):
    """Add Gaussian noise scaled by sqrt(Fisher) on the target set,
    normalized per tensor (noise RMS = 0.5 x weight RMS at lam=1)."""
    Xt, yt = ds.target_train()
    F = model.fisher_diag(Xt, yt)
    rng = np.random.default_rng(seed)
    m = model.copy()
    for P, Fp in zip(m.params(), F):
        scale = lam * np.sqrt(Fp)
        rms = np.sqrt((P ** 2).mean()) + 1e-12
        noise = rng.normal(size=P.shape) * scale
        nrms = np.sqrt((noise ** 2).mean()) + 1e-12
        P += noise * (rms / nrms) * 0.5
    return m


# ----------------------------------------------------------------------
# D. The keyed plate
# ----------------------------------------------------------------------
class KeyedPlate:
    """Private knowledge encoded through a key-derived codebook.

    - Private input slots are permuted by pi_in(key).
    - Private output codes are permuted by pi_out(key).

    The underlying MLP only ever sees permuted private data, so its
    weights never contain a plaintext question->answer mapping. Generic
    inputs/outputs are NOT keyed (they are public knowledge). Destroying
    the key destroys the mapping between plaintext facts and weight
    space; the weights that remain are, to every probe in this tool,
    junk with respect to the private facts.
    """

    def __init__(self, key_bytes, priv_dim, n_classes):
        seed = int.from_bytes(key_bytes[:8], "little") % (2 ** 31)
        rng = np.random.default_rng(seed)
        self.pi_in = rng.permutation(priv_dim)
        self.pi_out = rng.permutation(n_classes)
        self._inv_out = np.empty_like(self.pi_out)
        self._inv_out[self.pi_out] = np.arange(len(self.pi_out))
        self.gen_dim = None  # set by train_keyed / enc_in callers

    def enc_in(self, X, gen_dim):
        X = X.copy()
        X[:, gen_dim:] = X[:, gen_dim:][:, self.pi_in]
        return X

    def enc_labels_private(self, y):
        return self.pi_out[y]

    def dec_out(self, classes):
        return self._inv_out[classes]


def keyed_fact_acc(ds, model, plate, probe, facts):
    X, y, f = probe
    m = np.isin(f, facts)
    if m.sum() == 0:
        return float("nan")
    pred = plate.dec_out(model.predict(plate.enc_in(X[m], ds.gen_dim)))
    return float((pred == y[m]).mean())


def train_keyed(ds, scale, seed_init, key_bytes):
    """Train a from-birth keyed model. Returns (model, plate)."""
    plate = KeyedPlate(key_bytes, ds.priv_dim, ds.n_classes)
    model = MLP(ds.in_dim, scale.hidden, ds.n_classes, seed_init, lr=scale.lr)
    Xg, yg = ds.gX, ds.gy * ds.surfaces
    Xp = plate.enc_in(ds.pX, ds.gen_dim)
    yp = plate.enc_labels_private(ds.py)
    X = np.vstack([Xg, Xp])
    y = np.concatenate([yg, yp])
    steps = 0
    while steps < scale.max_train_epochs:
        model.train_batch(X, y, epochs=scale.train_block)
        steps += scale.train_block
        if generic_acc(model.predict, ds.gX_test, ds.gy_test,
                       ds.surfaces) < 0.95:
            continue
        if keyed_fact_acc(ds, model, plate, ds.probe, ds.controls) \
                >= scale.stop_acc:
            break
    return model, plate

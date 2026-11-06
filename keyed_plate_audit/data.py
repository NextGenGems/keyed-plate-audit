"""Dataset construction.

Mirrors the verified prototype's build_data() exactly — same RNG stream,
same construction order — generalized only in that facts come from a
FactSpec instead of index arithmetic. With the default spec, the arrays
produced are identical to the prototype's.

Layout of an input vector (float32):
  [0 : gen_dim]            generic block: one-hot concept x noisy prototype
  [gen_dim : in_dim]       private block: one slot per (fact, surface)

A fact's answer class is concept * surfaces + code, where `code` is the
private offset only the private slot determines.
"""

import numpy as np


class Dataset:
    def __init__(self, spec, scale, seed_data):
        self.spec = spec
        self.scale = scale
        self.n_concepts = spec.n_concepts
        self.surfaces = scale.surfaces
        self.conc_dim = scale.conc_dim
        self.n_facts = len(spec.facts)
        self.gen_dim = self.n_concepts * self.conc_dim
        self.priv_dim = self.n_facts * self.surfaces
        self.in_dim = self.gen_dim + self.priv_dim
        self.n_classes = self.n_concepts * self.surfaces
        self.targets = np.array(spec.target_indices())
        self.controls = np.array(spec.control_indices())
        self._build(seed_data)

    # -- construction (order matters: it fixes the RNG stream) ---------
    def _build(self, seed_data):
        rng = np.random.default_rng(seed_data)
        s = self.scale
        protos = rng.normal(0, 1.0, size=(self.n_concepts, self.conc_dim))

        def onehot(idx, n):
            v = np.zeros(n, dtype=np.float32)
            v[idx] = 1.0
            return v

        def generic_vec(concept):
            v = np.zeros(self.in_dim)
            v[:self.gen_dim] = np.kron(
                onehot(concept, self.n_concepts),
                protos[concept] + 0.15 * rng.normal(size=self.conc_dim))
            return v

        def private_vec(fact_idx, surface, jitter=0.05):
            concept = self.spec.facts[fact_idx].concept
            v = np.zeros(self.in_dim)
            v[:self.gen_dim] = np.kron(
                onehot(concept, self.n_concepts),
                protos[concept] + jitter * rng.normal(size=self.conc_dim))
            v[self.gen_dim + fact_idx * self.surfaces + surface] = 1.0
            return v

        def fact_target(fact_idx):
            f = self.spec.facts[fact_idx]
            return f.concept * self.surfaces + f.code

        gX, gy = [], []
        for _ in range(s.n_gen_ex):
            c = int(rng.integers(self.n_concepts))
            gX.append(generic_vec(c))
            gy.append(c)
        self.gX, self.gy = np.array(gX), np.array(gy)

        gXt, gyt = [], []
        for _ in range(s.gen_test_ex):
            c = int(rng.integers(self.n_concepts))
            gXt.append(generic_vec(c))
            gyt.append(c)
        self.gX_test, self.gy_test = np.array(gXt), np.array(gyt)

        pX, py, pf = [], [], []
        for fi in range(self.n_facts):
            for surf in range(self.surfaces):
                for _ in range(s.ex_per_fact):
                    pX.append(private_vec(fi, surf, jitter=0.08))
                    py.append(fact_target(fi))
                    pf.append(fi)
        self.pX, self.py, self.pf = np.array(pX), np.array(py), np.array(pf)

        def probe_set(surface_shift):
            X, y, f = [], [], []
            for fi in range(self.n_facts):
                base_surface = fi % self.surfaces
                surf = (base_surface + surface_shift) % self.surfaces
                for _ in range(s.probe_copies):
                    X.append(private_vec(fi, surf, jitter=0.08))
                    y.append(fact_target(fi))
                    f.append(fi)
            return np.array(X), np.array(y), np.array(f)

        probe = probe_set(0)        # trained surfaces, fresh noise
        paraphrase = probe_set(1)   # shifted surface = "paraphrase"

        f32 = lambda a: a.astype(np.float32)
        self.gX, self.gX_test = f32(self.gX), f32(self.gX_test)
        self.pX = f32(self.pX)
        self.probe = (f32(probe[0]), probe[1], probe[2])
        self.paraphrase = (f32(paraphrase[0]), paraphrase[1], paraphrase[2])

    # -- training batches ----------------------------------------------
    def mixed_batch(self):
        """Generic examples (label = concept * surfaces) + all private
        fact examples — the plaintext model's full training set."""
        X = np.vstack([self.gX, self.pX])
        y = np.concatenate([self.gy * self.surfaces, self.py])
        return X, y

    def attack_batch(self):
        """What the relearning attacker may train on: generic examples +
        CONTROL facts only. Never the targets."""
        cmask = np.isin(self.pf, self.controls)
        X = np.vstack([self.gX, self.pX[cmask]])
        y = np.concatenate([self.gy * self.surfaces, self.py[cmask]])
        return X, y

    def target_train(self):
        """Target-fact training rows (for Fisher importance in A–C)."""
        tmask = np.isin(self.pf, self.targets)
        return self.pX[tmask], self.py[tmask]

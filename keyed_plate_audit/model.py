"""A small numpy MLP (Adam, softmax cross-entropy) — the prototype's
model, unchanged except that dimensions come from the Dataset."""

import numpy as np


class MLP:
    def __init__(self, in_dim, hidden, n_classes, seed, lr=0.01):
        rng = np.random.default_rng(seed)
        self.in_dim, self.hidden, self.n_classes = in_dim, hidden, n_classes
        self.lr = lr
        self.W1 = rng.normal(0, np.sqrt(2.0 / in_dim),
                             (in_dim, hidden)).astype(np.float32)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = rng.normal(0, np.sqrt(2.0 / hidden),
                             (hidden, n_classes)).astype(np.float32)
        self.b2 = np.zeros(n_classes, dtype=np.float32)
        self._reset_adam()

    def _reset_adam(self):
        self.t = 0
        self.m = [np.zeros_like(p) for p in self.params()]
        self.v = [np.zeros_like(p) for p in self.params()]

    def params(self):
        return [self.W1, self.b1, self.W2, self.b2]

    def copy(self):
        other = MLP.__new__(MLP)
        other.in_dim, other.hidden = self.in_dim, self.hidden
        other.n_classes, other.lr = self.n_classes, self.lr
        other.W1, other.b1 = self.W1.copy(), self.b1.copy()
        other.W2, other.b2 = self.W2.copy(), self.b2.copy()
        other._reset_adam()
        return other

    def forward(self, X):
        H = np.tanh(X @ self.W1 + self.b1)
        Z = H @ self.W2 + self.b2
        Z = Z - Z.max(axis=1, keepdims=True)
        E = np.exp(Z)
        P = E / E.sum(axis=1, keepdims=True)
        return H, P

    def predict(self, X):
        return self.forward(X)[1].argmax(axis=1)

    def train_batch(self, X, y, lr=None, epochs=1):
        """`epochs` full-batch gradient updates. One update = one
        'gradient step' in all step accounting in this tool."""
        if lr is None:
            lr = self.lr
        n = len(X)
        Y = np.zeros((n, self.n_classes))
        Y[np.arange(n), y] = 1.0
        for _ in range(epochs):
            H, P = self.forward(X)
            dZ = (P - Y) / n
            gW2 = H.T @ dZ
            gb2 = dZ.sum(axis=0)
            dH = (dZ @ self.W2.T) * (1.0 - H ** 2)
            gW1 = X.T @ dH
            gb1 = dH.sum(axis=0)
            grads = [gW1, gb1, gW2, gb2]
            self.t += 1
            ps = self.params()
            for i in range(4):
                self.m[i] = 0.9 * self.m[i] + 0.1 * grads[i]
                self.v[i] = 0.999 * self.v[i] + 0.001 * grads[i] ** 2
                mh = self.m[i] / (1 - 0.9 ** self.t)
                vh = self.v[i] / (1 - 0.999 ** self.t)
                ps[i] -= lr * mh / (np.sqrt(vh) + 1e-8)
            self.W1, self.b1, self.W2, self.b2 = ps

    def fisher_diag(self, X, y):
        """Empirical diagonal Fisher: mean over examples of the SQUARED
        per-example score (not the square of the mean gradient — that
        vanishes on a fitted model and would make ablation meaningless)."""
        H, P = self.forward(X)
        n = len(X)
        Y = np.zeros((n, self.n_classes))
        Y[np.arange(n), y] = 1.0
        dZ = (P - Y)
        dH = (dZ @ self.W2.T) * (1.0 - H ** 2)
        FW2 = (H ** 2).T @ (dZ ** 2) / n
        Fb2 = (dZ ** 2).mean(axis=0)
        FW1 = (X ** 2).T @ (dH ** 2) / n
        Fb1 = (dH ** 2).mean(axis=0)
        return [FW1, Fb1, FW2, Fb2]


# ----------------------------------------------------------------------
# Evaluation helpers
# ----------------------------------------------------------------------
def fact_acc(model_fn, probe, facts):
    """Accuracy of model_fn on the probe rows belonging to `facts`."""
    X, y, f = probe
    m = np.isin(f, facts)
    if m.sum() == 0:
        return float("nan")
    pred = model_fn(X[m])
    return float((pred == y[m]).mean())


def generic_acc(model_fn, X, y_concepts, surfaces):
    """Generic task = name the concept. Model outputs classes
    (concept * surfaces + code), so predictions are mapped back to
    concepts before comparing."""
    return float((model_fn(X) // surfaces == y_concepts).mean())

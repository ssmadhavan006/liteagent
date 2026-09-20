"""
Fits the complexity scorer against capability-grounded labels.

The shipped weights were hand-tuned to reproduce a rubric defined in terms of
prompt length, which is also the scorer's dominant feature. Measured against
labels derived from what the models can actually solve, that configuration
scores below a constant predictor, so the parameters are refit here against the
capability labels instead.

Model form matches inference exactly. `ComplexityScorer` computes
`s = sigmoid(w.x + b)` and assigns a tier by comparing `s` to `theta_low` and
`theta_high`. Because sigmoid is monotonic, that is an ordinal model with two
cutpoints on the linear predictor, so ordinal logistic regression is the
matching estimator and its cutpoints convert straight back into thresholds:

    bias        = 0
    theta_low   = sigmoid(c0)
    theta_high  = sigmoid(c1)
"""

import json
import math

import numpy as np

FEATURE_ORDER = [
    "length_chars",
    "code_syntax_count",
    "math_operator_density",
    "math_keyword_count",
    "logical_count",
    "question_count",
    "entity_density",
]

TIERS = ["Low", "Medium", "High"]


def featurise(prompt: str) -> np.ndarray:
    from liteagent.router.features import extract_raw_features, normalize_features
    feats = normalize_features(extract_raw_features(prompt))
    return np.array([feats.get(name, 0.0) for name in FEATURE_ORDER], dtype=np.float64)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def fit_ordinal(
    X: np.ndarray,
    y: np.ndarray,
    l2: float = 1.0,
    lr: float = 0.5,
    epochs: int = 4000,
    seed: int = 42,
) -> tuple[np.ndarray, float, float]:
    """
    Ordinal logistic regression with two cutpoints.

    P(y=0) = sig(c0 - z), P(y=1) = sig(c1 - z) - sig(c0 - z), P(y=2) = 1 - sig(c1 - z)
    where z = w.x. Cutpoints are kept ordered by optimising an unconstrained gap.

    L2 regularisation is deliberately strong: with ~60 training rows and seven
    correlated surface features, an unpenalised fit memorises the split.
    """
    rng = np.random.default_rng(seed)
    n, d = X.shape
    w = rng.normal(0.0, 0.01, size=d)
    c0 = -0.5
    raw_gap = 0.0  # c1 = c0 + softplus(raw_gap), keeping c1 > c0

    for _ in range(epochs):
        gap = math.log1p(math.exp(min(raw_gap, 30)))
        c1 = c0 + gap
        z = X @ w
        p0 = _sigmoid(c0 - z)
        p1 = _sigmoid(c1 - z)

        prob = np.empty((n, 3))
        prob[:, 0] = p0
        prob[:, 1] = np.maximum(p1 - p0, 1e-9)
        prob[:, 2] = np.maximum(1.0 - p1, 1e-9)
        prob = np.clip(prob, 1e-9, 1.0)

        # d(NLL)/dz and d/dc for the observed class.
        g_w = np.zeros(d)
        g_c0 = 0.0
        g_gap = 0.0
        for i in range(n):
            k = y[i]
            a0 = p0[i] * (1 - p0[i])
            a1 = p1[i] * (1 - p1[i])
            if k == 0:
                dz = a0 / prob[i, 0]
                dc0 = -a0 / prob[i, 0]
                dc1 = 0.0
            elif k == 1:
                dz = (a1 - a0) / prob[i, 1]
                dc0 = a0 / prob[i, 1]
                dc1 = -a1 / prob[i, 1]
            else:
                dz = -a1 / prob[i, 2]
                dc0 = 0.0
                dc1 = a1 / prob[i, 2]
            g_w += dz * X[i]
            g_c0 += dc0 + dc1
            g_gap += dc1

        g_w = g_w / n + l2 * w / n
        sig_gap = 1.0 / (1.0 + math.exp(-min(max(raw_gap, -30), 30)))
        w -= lr * g_w
        c0 -= lr * (g_c0 / n)
        raw_gap -= lr * (g_gap / n) * sig_gap

    gap = math.log1p(math.exp(min(raw_gap, 30)))
    return w, c0, c0 + gap


def predict(X: np.ndarray, w: np.ndarray, c0: float, c1: float) -> np.ndarray:
    z = X @ w
    out = np.full(len(z), 1, dtype=int)
    out[z < c0] = 0
    out[z >= c1] = 2
    return out


def stratified_split(y: np.ndarray, test_frac: float = 0.35, seed: int = 42):
    """Keeps the tier mix of the full set in both halves."""
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        cut = max(1, int(round(len(idx) * test_frac)))
        test_idx.extend(idx[:cut])
        train_idx.extend(idx[cut:])
    return np.array(sorted(train_idx)), np.array(sorted(test_idx))


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    p = successes / total
    denom = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    return max(0.0, centre - margin), min(1.0, centre + margin)


def to_config(w: np.ndarray, c0: float, c1: float) -> dict:
    """Converts fitted parameters into the router_config.yaml layout."""
    return {
        "weights": {name: float(round(wi, 4)) for name, wi in zip(FEATURE_ORDER, w)},
        "bias": 0.0,
        "routing": {
            "theta_low": float(round(1.0 / (1.0 + math.exp(-c0)), 4)),
            "theta_high": float(round(1.0 / (1.0 + math.exp(-c1)), 4)),
        },
    }


def load_capability_dataset(path: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Loads labels, strips the constant protocol prefix, and featurises."""
    from tests.router.analyze_labels import strip_protocol_prefix

    X, y, benchmarks = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec["capability_tier"] == "Unsolved":
                continue
            body = strip_protocol_prefix(rec["prompt"], rec["benchmark"])
            X.append(featurise(body))
            y.append(TIERS.index(rec["capability_tier"]))
            benchmarks.append(rec["benchmark"])
    return np.array(X), np.array(y), benchmarks

"""
Refits the complexity scorer against capability-grounded labels and reports
held-out accuracy against the baselines that matter.

Usage:
    uv run python -m tests.router.calibrate_router
    uv run python -m tests.router.calibrate_router --write   # update the YAML

The comparison to beat is the majority-class predictor. A router that cannot
beat "always predict the most common tier" carries no routing signal, however
sophisticated its features are.
"""

import argparse
import json
from collections import Counter

import numpy as np

from liteagent.router.calibration import (
    FEATURE_ORDER,
    TIERS,
    fit_ordinal,
    load_capability_dataset,
    predict,
    stratified_split,
    to_config,
    wilson,
)

LABELS = "datasets/router_validation/capability_labels.jsonl"
CONFIG = "config/router_config.yaml"


def report(name: str, correct: int, total: int) -> float:
    acc = correct / total * 100 if total else 0.0
    lo, hi = wilson(correct, total)
    print(f"  {name:34} {acc:6.2f}%  (95% CI {lo*100:.1f}-{hi*100:.1f}%, n={total})")
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=LABELS)
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--test-frac", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--write", action="store_true", help="Write fitted params into the router YAML")
    args = ap.parse_args()

    X, y, benchmarks = load_capability_dataset(args.labels)
    print(f"Capability-labelled tasks: {len(y)}")
    print("Distribution:", {TIERS[k]: v for k, v in sorted(Counter(y).items())})

    tr, te = stratified_split(y, args.test_frac, args.seed)
    print(f"Split: {len(tr)} train / {len(te)} test (stratified, seed {args.seed})\n")

    w, c0, c1 = fit_ordinal(X[tr], y[tr], l2=args.l2, seed=args.seed)

    # Baselines computed on the training split, evaluated on the held-out split.
    majority_cls = Counter(y[tr]).most_common(1)[0][0]

    print("=== Held-out accuracy (capability labels) ===")
    fitted_correct = int((predict(X[te], w, c0, c1) == y[te]).sum())
    majority_correct = int((y[te] == majority_cls).sum())

    report(f"majority class ('{TIERS[majority_cls]}')", majority_correct, len(te))
    fitted_acc = report("refit ordinal router", fitted_correct, len(te))

    # The shipped hand-tuned configuration, for reference.
    from liteagent.router.classifier import ComplexityScorer
    from liteagent.router.pruning import map_tier_and_pruning
    scorer = ComplexityScorer(CONFIG)
    shipped_correct = 0
    for i in te:
        feats = {name: X[i][j] for j, name in enumerate(FEATURE_ORDER)}
        score, _ = scorer.score_task(feats)
        tier, _, _, _ = map_tier_and_pruning(score, scorer.theta_low, scorer.theta_high,
                                             benchmarks[i])
        if TIERS.index({"Small": "Low", "Medium": "Medium", "Large": "High"}[tier]) == y[i]:
            shipped_correct += 1
    report("shipped hand-tuned config", shipped_correct, len(te))

    maj_lo, maj_hi = wilson(majority_correct, len(te))
    fit_lo, _ = wilson(fitted_correct, len(te))
    print()
    if fit_lo > maj_hi:
        print("  VERDICT: the refit router beats majority class with non-overlapping CIs.")
    elif fitted_correct > majority_correct:
        print("  VERDICT: refit is nominally ahead of majority class, but the confidence")
        print("           intervals overlap - this split cannot establish a real gain.")
    else:
        print("  VERDICT: the refit router does NOT beat majority class. Surface features")
        print("           do not carry usable signal about which model can solve a task.")

    print("\n=== Fitted parameters ===")
    cfg = to_config(w, c0, c1)
    print(json.dumps(cfg, indent=2))

    if args.write:
        import yaml
        with open(CONFIG, "r", encoding="utf-8") as f:
            existing = yaml.safe_load(f) or {}
        existing["version"] = int(existing.get("version", 1)) + 1
        existing["weights"] = cfg["weights"]
        existing["bias"] = cfg["bias"]
        existing.setdefault("routing", {}).update(cfg["routing"])
        with open(CONFIG, "w", encoding="utf-8") as f:
            yaml.safe_dump(existing, f, sort_keys=False)
        print(f"\nWrote fitted parameters to {CONFIG} (version {existing['version']})")


if __name__ == "__main__":
    main()

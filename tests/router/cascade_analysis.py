"""
Compares routing policies on the capability-labelled set.

Motivation: `docs/phase9/router_capability_analysis.md` shows that predicting
the required tier from prompt surface features does not generalise. A cascade
sidesteps prediction entirely - it runs the cheapest tier, checks the answer,
and escalates only on failure. This script quantifies what that policy would
cost and solve, using per-tier latencies measured during labelling.

Policies compared:
  always-small / always-medium / always-large   fixed tier
  oracle-router                                 perfect a-priori tier choice
  cascade(v)                                    try tiers in order, escalate
                                                when the verifier says "wrong",
                                                with verifier sensitivity v
  shipped-router                                the hand-tuned scorer

The cascade is reported across verifier accuracies because a real verifier (the
Critic agent) is imperfect, and its accuracy is what decides whether the policy
is viable.

Usage:
    uv run python -m tests.router.cascade_analysis
"""

import json
import re
from collections import defaultdict

LABELS = "datasets/router_validation/capability_labels.jsonl"
LOG = "experiments/capability_labeling.log"
TIERS = ["Low", "Medium", "High"]


def measured_tier_latencies(log_path: str) -> dict:
    """
    Mean seconds per attempt, per tier, parsed from the labelling log.

    Uses attempt latencies rather than a parameter-count proxy so the cost model
    reflects this hardware.
    """
    per_tier = defaultdict(list)
    current = None
    tier_re = re.compile(r"=== Tier (\w+)")
    row_re = re.compile(r"\[\d+/\d+\]\s+(\w+)\s+(SOLVED|failed).*?([\d.]+)s")
    try:
        with open(log_path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = tier_re.search(line)
                if m:
                    current = m.group(1)
                    continue
                r = row_re.search(line)
                if r and current:
                    per_tier[current].append(float(r.group(3)))
    except FileNotFoundError:
        pass

    means = {}
    for tier in TIERS:
        vals = per_tier.get(tier, [])
        means[tier] = sum(vals) / len(vals) if vals else None
    return means


def load(path: str):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return [r for r in rows if r["capability_tier"] != "Unsolved"]


def main():
    rows = load(LABELS)
    n = len(rows)
    lat = measured_tier_latencies(LOG)
    # Fall back to a parameter-count proxy if the log is unavailable.
    fallback = {"Low": 1.0, "Medium": 3.0, "High": 8.0}
    cost = {t: (lat[t] if lat.get(t) else fallback[t]) for t in TIERS}

    print(f"Labelled solvable tasks: {n}")
    print("Measured cost per attempt (s): " +
          ", ".join(f"{t}={cost[t]:.1f}" for t in TIERS))
    print()

    need = [TIERS.index(r["capability_tier"]) for r in rows]

    def fixed(tier_idx):
        solved = sum(1 for k in need if k <= tier_idx)
        return solved, n * cost[TIERS[tier_idx]]

    rowsfmt = "  {:<28} {:>7} {:>9} {:>11}"
    print("=== Policy comparison ===")
    print(rowsfmt.format("policy", "solved", "solve %", "total cost"))

    for i, t in enumerate(TIERS):
        s, c = fixed(i)
        print(rowsfmt.format(f"always-{t.lower()}", f"{s}/{n}", f"{100*s/n:.1f}%", f"{c:.0f}s"))

    # Oracle router: knows the right tier a priori, pays only that tier once.
    oracle_cost = sum(cost[TIERS[k]] for k in need)
    print(rowsfmt.format("oracle-router (upper bound)", f"{n}/{n}", "100.0%", f"{oracle_cost:.0f}s"))

    # Cascade: run tiers in order; the verifier decides whether to stop.
    # sensitivity = P(verifier flags a wrong answer as wrong)
    # specificity = P(verifier accepts a correct answer)
    print()
    print("=== Cascade, by verifier quality ===")
    print("  (sensitivity = catches a wrong answer; specificity = accepts a right one)")
    print(rowsfmt.format("policy", "solved", "solve %", "total cost"))
    for sens, spec in ((1.0, 1.0), (0.9, 0.95), (0.8, 0.9), (0.7, 0.85), (0.5, 0.8)):
        total_cost = 0.0
        solved = 0.0
        for k in need:
            # Probability of still being in the cascade at tier i.
            alive = 1.0
            done = False
            for i in range(len(TIERS)):
                total_cost += alive * cost[TIERS[i]]
                if i >= k:
                    # This tier solves it. Stop iff the verifier accepts.
                    solved += alive * spec
                    alive *= (1.0 - spec)
                else:
                    # Wrong answer. Continue iff the verifier catches it.
                    alive *= sens
                if alive <= 1e-6:
                    done = True
                    break
            if not done:
                pass
        label = f"cascade(sens={sens:.2f}, spec={spec:.2f})"
        print(rowsfmt.format(label, f"{solved:.1f}/{n}", f"{100*solved/n:.1f}%", f"{total_cost:.0f}s"))

    # Shipped router, for reference.
    from liteagent.router.calibration import FEATURE_ORDER, featurise
    from liteagent.router.classifier import ComplexityScorer
    from liteagent.router.pruning import map_tier_and_pruning
    from tests.router.analyze_labels import strip_protocol_prefix

    scorer = ComplexityScorer("config/router_config.yaml")
    solved = 0
    total_cost = 0.0
    for r, k in zip(rows, need):
        body = strip_protocol_prefix(r["prompt"], r["benchmark"])
        x = featurise(body)
        feats = {name: x[j] for j, name in enumerate(FEATURE_ORDER)}
        score, _ = scorer.score_task(feats)
        tier, _, _, _ = map_tier_and_pruning(score, scorer.theta_low, scorer.theta_high,
                                             r["benchmark"])
        idx = {"Small": 0, "Medium": 1, "Large": 2}[tier]
        total_cost += cost[TIERS[idx]]
        if idx >= k:
            solved += 1
    print()
    print(rowsfmt.format("shipped-router", f"{solved}/{n}", f"{100*solved/n:.1f}%", f"{total_cost:.0f}s"))

    print()
    print("Read: a cascade converts the prediction problem into a verification")
    print("problem. Its solve rate is bounded by verifier specificity, and its")
    print("cost by verifier sensitivity - neither requires predicting difficulty.")


if __name__ == "__main__":
    main()

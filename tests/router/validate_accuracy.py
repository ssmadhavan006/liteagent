import json
import math
import os
import sys
from liteagent.router.router import route_task

def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """
    Wilson score interval for a binomial proportion.

    Preferred over the normal approximation here because the held-out split is
    small enough (n=30) that the naive interval would run past 0/1 and understate
    the uncertainty.
    """
    if total == 0:
        return 0.0, 0.0
    p = successes / total
    denom = 1.0 + (z ** 2) / total
    center = (p + (z ** 2) / (2 * total)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / total + (z ** 2) / (4 * total ** 2))
    return max(0.0, center - margin), min(1.0, center + margin)


def majority_class_baseline(prompts: list) -> float:
    """Accuracy a trivial always-predict-the-most-common-tier router would get."""
    tier_mapping = {"Low": "Small", "Medium": "Medium", "High": "Large"}
    counts = {}
    for item in prompts:
        tier = tier_mapping.get(item["expected_tier"], item["expected_tier"])
        counts[tier] = counts.get(tier, 0) + 1
    if not counts:
        return 0.0
    return max(counts.values()) / len(prompts) * 100.0


def is_adjacent(expected, predicted):
    adjacent_pairs = {
        ("Small", "Medium"), ("Medium", "Small"),
        ("Medium", "Large"), ("Large", "Medium")
    }
    return (expected, predicted) in adjacent_pairs

def run_validation():
    dataset_path = "datasets/router_validation/validation_prompts.json"
    config_path = "config/router_config.yaml"

    if not os.path.exists(dataset_path):
        print(f"Error: Validation dataset not found at {dataset_path}")
        sys.exit(1)

    with open(dataset_path, "r") as f:
        prompts = json.load(f)

    # Strictly filter for held-out test set (IDs 51 to 80)
    test_prompts = [p for p in prompts if p["id"] > 50]
    total = len(test_prompts)

    matches = 0
    mismatches = []

    adjacent_mismatches = 0
    far_mismatches = 0

    log_dir = "experiments/validation"
    log_file = os.path.join(log_dir, "routing_decisions.jsonl")
    if os.path.exists(log_file):
        try:
            os.remove(log_file)
        except OSError:
            pass

    print(f"Running validation on {total} held-out test prompts (IDs 51-80)...")
    print("-" * 60)

    for item in test_prompts:
        task = {
            "prompt": item["prompt"],
            "benchmark": item["benchmark"],
            "metadata": {"original_id": item["id"]}
        }

        result = route_task(task, config_path=config_path, log_dir=log_dir)
        pred_tier = result["model_tier"]
        exp_tier = item["expected_tier"]

        tier_mapping = {"Low": "Small", "Medium": "Medium", "High": "Large"}
        mapped_exp_tier = tier_mapping.get(exp_tier, exp_tier)

        if pred_tier == mapped_exp_tier:
            matches += 1
        else:
            mismatch_type = "Adjacent" if is_adjacent(mapped_exp_tier, pred_tier) else "Far"
            if mismatch_type == "Adjacent":
                adjacent_mismatches += 1
            else:
                far_mismatches += 1

            mismatches.append({
                "id": item["id"],
                "benchmark": item["benchmark"],
                "prompt_snippet": item["prompt"][:60] + "...",
                "expected": mapped_exp_tier,
                "predicted": pred_tier,
                "score": round(result["metadata"]["score"], 4),
                "margin": round(result["metadata"]["routing_margin"], 4),
                "type": mismatch_type,
                "reasons": result["metadata"]["decision_reasons"]
            })

    accuracy = (matches / total) * 100.0
    ci_low, ci_high = wilson_interval(matches, total)
    majority = majority_class_baseline(test_prompts)
    random_chance = 100.0 / 3.0

    print("Validation Complete (Test Split).")
    print(f"Total Test Prompts: {total}")
    print(f"Correctly Routed: {matches}")
    print(f"Mismatched: {len(mismatches)}")
    print(f"  - Adjacent-Tier Errors: {adjacent_mismatches}")
    print(f"  - Far-Tier Errors: {far_mismatches}")
    print(f"Held-Out Test Accuracy: {accuracy:.2f}% "
          f"(95% Wilson CI: {ci_low*100:.2f}%-{ci_high*100:.2f}%, n={total})")
    print(f"  Reference points: random={random_chance:.2f}%, majority-class={majority:.2f}%")
    if ci_low * 100.0 <= majority:
        print("  NOTE: the confidence interval includes the majority-class baseline. "
              "This split is too small to claim the router beats it.")
    print("-" * 60)

    if mismatches:
        print("Mismatch Details & Error Analysis:")
        for m in mismatches:
            print(f"ID {m['id']} [{m['benchmark']}]: Expected {m['expected']} | Predicted {m['predicted']} ({m['type']} Error)")
            print(f"  Snippet: {m['prompt_snippet']}")
            print(f"  Score: {m['score']} | Margin: {m['margin']} | Reasons: {m['reasons']}")
            print()
    else:
        print("All test prompts routed perfectly to their expected tiers!")

    return accuracy

if __name__ == "__main__":
    run_validation()

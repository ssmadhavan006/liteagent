"""
Compares the capability-grounded labels against the original rubric labels and
against a length-only predictor.

The point of this report is to show whether the router is learning task
difficulty or merely re-deriving prompt length. If a length threshold beats the
router on a label set, that label set is measuring length.

Usage:
    uv run python -m tests.router.analyze_labels
"""

import json
import os
from collections import Counter

from liteagent.router.capability_labels import cohens_kappa
from liteagent.router.router import route_task

LABELS_PATH = "datasets/router_validation/capability_labels.jsonl"
CONFIG_PATH = "config/router_config.yaml"


def load_labels(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def strip_protocol_prefix(prompt: str, benchmark: str) -> str:
    """
    Recovers the bare task text from a stored prompt.

    Stored prompts include the output contract and few-shot exemplars, which are
    byte-identical across every task in a benchmark. Routing on them would make
    all prompts look the same length and destroy the router's dominant feature,
    so the constant prefix is removed before scoring.
    """
    from liteagent.eval.fewshot import build_prefix
    from liteagent.eval.harness import BENCHMARK_SYSTEM_PROMPTS

    system = BENCHMARK_SYSTEM_PROMPTS.get(benchmark, "")
    prefix = build_prefix(benchmark)
    body = prompt
    if system and body.startswith(system):
        body = body[len(system):].lstrip("\n")
    if prefix and body.startswith(prefix):
        body = body[len(prefix):]
    return body


def fit_length_thresholds(records: list[dict]) -> tuple[int, int]:
    """Best two-threshold length classifier, fit on the given records."""
    best = (-1, (60, 190))
    for lo in range(20, 1000, 10):
        for hi in range(lo + 20, 4000, 10):
            hits = sum(
                1 for r in records
                if (r["tier"] == "Low" and len(r["prompt"]) < lo)
                or (r["tier"] == "Medium" and lo <= len(r["prompt"]) < hi)
                or (r["tier"] == "High" and len(r["prompt"]) >= hi)
            )
            if hits > best[0]:
                best = (hits, (lo, hi))
    return best[1]


def length_predict(prompt: str, lo: int, hi: int) -> str:
    n = len(prompt)
    return "Low" if n < lo else ("Medium" if n < hi else "High")


def main():
    if not os.path.exists(LABELS_PATH):
        print(f"No capability labels at {LABELS_PATH}. Run build_capability_labels first.")
        return

    raw = load_labels(LABELS_PATH)
    records = [
        {"prompt": strip_protocol_prefix(r["prompt"], r["benchmark"]),
         "tier": r["capability_tier"],
         "benchmark": r["benchmark"], "external": r.get("external_difficulty")}
        for r in raw if r["capability_tier"] != "Unsolved"
    ]
    unsolved = len(raw) - len(records)

    print(f"Capability-labelled tasks: {len(records)} usable, {unsolved} unsolved by any tier")
    print("Distribution:", dict(Counter(r["tier"] for r in records)))
    print("By benchmark:", dict(Counter(f"{r['benchmark']}/{r['tier']}" for r in records)))

    if not records:
        return

    # Router vs capability labels
    router_preds, truth = [], []
    for r in records:
        res = route_task({"prompt": r["prompt"], "benchmark": r["benchmark"]},
                         config_path=CONFIG_PATH, log_dir=None)
        router_preds.append(res["model_tier"])
        truth.append({"Low": "Small", "Medium": "Medium", "High": "Large"}[r["tier"]])

    router_acc = sum(p == t for p, t in zip(router_preds, truth)) / len(truth) * 100

    # Length-only predictor, fit on the same data (optimistic upper bound for it)
    lo, hi = fit_length_thresholds(records)
    len_preds = [
        {"Low": "Small", "Medium": "Medium", "High": "Large"}[length_predict(r["prompt"], lo, hi)]
        for r in records
    ]
    len_acc = sum(p == t for p, t in zip(len_preds, truth)) / len(truth) * 100

    majority = max(Counter(truth).values()) / len(truth) * 100

    print("\n=== Against capability-grounded labels ===")
    print(f"  LiteAgent router : {router_acc:.2f}%")
    print(f"  Length-only (in-sample, thresholds {lo}/{hi}) : {len_acc:.2f}%")
    print(f"  Majority class   : {majority:.2f}%")
    if len_acc > router_acc:
        print("  NOTE: length still beats the router. Either the router needs "
              "recalibration on these labels, or length remains confounded.")
    else:
        print("  The router beats a length-only predictor on non-circular labels.")

    # Agreement between the original rubric labels and measured capability
    old_path = "datasets/router_validation/validation_prompts.json"
    if os.path.exists(old_path):
        old = json.load(open(old_path, encoding="utf-8"))
        by_prompt = {o["prompt"]: o["expected_tier"] for o in old}
        pairs = [(by_prompt[r["prompt"]], r["tier"]) for r in records if r["prompt"] in by_prompt]
        if pairs:
            k = cohens_kappa([a for a, _ in pairs], [b for _, b in pairs])
            print(f"\n=== Rubric vs capability agreement (n={len(pairs)}) ===")
            print(f"  Cohen's kappa: {k:.3f}")

    # HotpotQA ships its own human difficulty labels - a genuine second annotator
    ext = [(r["external"], r["tier"]) for r in records if r.get("external")]
    if ext:
        print(f"\n=== HotpotQA external human difficulty vs measured capability (n={len(ext)}) ===")
        print(" ", dict(Counter(f"{e}->{t}" for e, t in ext)))
        print("  (HotpotQA 'level' is annotated by the dataset authors, independent of us.)")


if __name__ == "__main__":
    main()

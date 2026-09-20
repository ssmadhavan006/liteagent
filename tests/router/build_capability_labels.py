"""
Builds a capability-grounded router validation set.

Usage:
    uv run python -m tests.router.build_capability_labels --per-benchmark 40

Writes JSONL incrementally so a long CPU run can be interrupted and resumed.
"""

import argparse
import json
import os

from liteagent.router.capability_labels import load_candidates, run_capability_pass


def main():
    parser = argparse.ArgumentParser(description="Build capability-grounded router labels")
    parser.add_argument("--per-benchmark", type=int, default=40,
                        help="Candidate prompts drawn per benchmark")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--out", type=str,
                        default="datasets/router_validation/capability_labels.jsonl")
    parser.add_argument("--max-tokens", type=int, default=320)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true",
                        help="Skip prompts already present in the output file")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    candidates = load_candidates(args.data_dir, args.per_benchmark, args.seed)

    if args.resume and os.path.exists(args.out):
        done = set()
        with open(args.out, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done.add(json.loads(line)["prompt"])
        before = len(candidates)
        candidates = [c for c in candidates if c["prompt"] not in done]
        print(f"Resuming: {before - len(candidates)} already labelled, {len(candidates)} remaining.")

    print(f"Labelling {len(candidates)} candidate prompts across three tiers.")
    records = run_capability_pass(candidates, args.out, max_tokens=args.max_tokens)

    from collections import Counter
    dist = Counter(r["capability_tier"] for r in records)
    print("\n=== Capability label distribution ===")
    for tier in ("Low", "Medium", "High", "Unsolved"):
        print(f"  {tier:9}: {dist.get(tier, 0)}")
    print(f"\nWrote {len(records)} records to {args.out}")


if __name__ == "__main__":
    main()

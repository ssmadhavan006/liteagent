"""
Measures the execution verifier against the same ground truth used for the Critic.

Reuses the answers already generated for the Critic measurement where possible,
so the two verifiers are compared on an identical set of candidate answers
rather than on separately sampled ones.

The hidden test suite is used **only to establish ground truth for this
measurement**. The verifier itself never sees it; it decides from the prompt's
worked examples alone.

Usage:
    uv run python -m tests.router.measure_exec_verifier
"""

import argparse
import gc
import json
import os
import time

from liteagent.eval.exec_verifier import extract_examples, verify_completion
from liteagent.eval.harness import BENCHMARK_MAX_TOKENS
from liteagent.router.capability_labels import _load_model, is_correct, load_candidates
from liteagent.utils.inference import InferenceEngine

OUT = "experiments/exec_verifier_measurement.jsonl"
LABELS = "datasets/router_validation/capability_labels.jsonl"
TIER_MODELS = {"Small": "llama3.2:1b", "Medium": "llama3.2:3b", "Large": "llama3.1:8b"}


def solvable_prompts(path: str) -> set[str]:
    keep = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    if rec["capability_tier"] != "Unsolved":
                        keep.add(rec["prompt"])
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="Small")
    ap.add_argument("--per-benchmark", type=int, default=40)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    keep = solvable_prompts(LABELS)

    cands = [c for c in load_candidates("data", args.per_benchmark, 42)
             if c["benchmark"] == "humaneval" and (not keep or c["prompt"] in keep)]
    print(f"HumanEval tasks: {len(cands)} (tier {args.tier})", flush=True)

    n_extractable = sum(
        1 for c in cands
        if extract_examples(c["prompt"], c["item"].get("entry_point", ""))
    )
    print(f"Prompts with executable examples: {n_extractable}/{len(cands)}\n", flush=True)

    rows = []
    llama = _load_model(TIER_MODELS[args.tier], n_ctx=4096)
    try:
        for i, cand in enumerate(cands, 1):
            item = cand["item"]
            budget = BENCHMARK_MAX_TOKENS.get("humaneval", 512)
            start = time.time()
            try:
                llama.reset()
                llama.eval(llama.tokenize(cand["prompt"].encode("utf-8")))
                _, completion = InferenceEngine.generate_tokens(llama, budget, temperature=0.0)
            except Exception as exc:
                print(f"  [{i}] generation error: {exc}", flush=True)
                completion = ""

            # Ground truth: the hidden suite. Measurement only.
            truth, _ = is_correct("humaneval", completion, item)
            # Verifier decision: prompt examples only.
            v = verify_completion(completion, item.get("prompt", ""),
                                  item.get("entry_point", ""))

            rows.append({
                "entry_point": item.get("entry_point", ""),
                "answer_correct": truth,
                "verdict": v["verdict"],
                "examples_run": v["examples_run"],
                "reason": v["reason"][:120],
            })
            with open(args.out, "a", encoding="utf-8") as f:
                f.write(json.dumps(rows[-1]) + "\n")
            print(f"  [{i}/{len(cands)}] {item.get('entry_point','')[:22]:22} "
                  f"truth={'OK ' if truth else 'bad'} verdict={v['verdict']:<12} "
                  f"{time.time()-start:.1f}s", flush=True)
    finally:
        try:
            llama.close()
        except Exception:
            pass
        del llama
        gc.collect()

    # Confusion matrix. `unverifiable` is reported separately rather than folded
    # into either outcome, since treating it as "pass" is exactly the failure
    # mode that made the LLM Critic useless.
    decided = [r for r in rows if r["verdict"] in ("pass", "fail")]
    tp = sum(1 for r in decided if not r["answer_correct"] and r["verdict"] == "fail")
    fn = sum(1 for r in decided if not r["answer_correct"] and r["verdict"] == "pass")
    tn = sum(1 for r in decided if r["answer_correct"] and r["verdict"] == "pass")
    fp = sum(1 for r in decided if r["answer_correct"] and r["verdict"] == "fail")
    unver = sum(1 for r in rows if r["verdict"] == "unverifiable")

    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")

    print("\n=== Execution verifier (HumanEval) ===")
    print(f"  decided: {len(decided)}/{len(rows)}   unverifiable: {unver}")
    print(f"  TP={tp} FN={fn} TN={tn} FP={fp}")
    print(f"  sensitivity = {sens:.3f}   specificity = {spec:.3f}   "
          f"Youden J = {sens + spec - 1:+.3f}")
    print("\n  Compare: LLM Critic on HumanEval @1B was sens=0.85 spec=0.09 (J=-0.06).")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()

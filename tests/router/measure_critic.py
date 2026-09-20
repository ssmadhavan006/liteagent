"""
H1b: measures the Critic's sensitivity and specificity.

The cascade replaces tier prediction with tier *observation*: run the cheap
model, ask the Critic whether the answer is right, escalate if it says no. Two
numbers decide whether that works, and neither has been measured.

    sensitivity = P(Critic rejects | answer is actually wrong)
        Low sensitivity means wrong answers are accepted and never escalated,
        so the cascade's solve rate collapses toward always-small.

    specificity = P(Critic approves | answer is actually correct)
        Low specificity means correct answers are rejected and escalated
        anyway, so the cost advantage over always-large disappears.

Ground truth comes from the benchmark metric, so "actually wrong" is decided by
the same scorer used in evaluation rather than by another model.

The Executor runs at the cascade's entry tier. The Critic is measured at that
tier and at the next one up, because "verify with a larger model" is a live
design option and its cost is only justified if it verifies better.

Usage:
    uv run python -m tests.router.measure_critic --limit 94
"""

import argparse
import gc
import json
import os
import time

from liteagent.agents.messages import Blackboard, CRITIQUE, DRAFT
from liteagent.agents.roles import CriticAgent, ExecutorAgent
from liteagent.eval.harness import BENCHMARK_MAX_TOKENS, BENCHMARK_SYSTEM_PROMPTS
from liteagent.router.capability_labels import (
    _load_model,
    is_correct,
    load_candidates,
)
from liteagent.utils.inference import InferenceEngine

OUT = "experiments/critic_measurement.jsonl"
LABELS = "datasets/router_validation/capability_labels.jsonl"
TIER_MODELS = {"Small": "llama3.2:1b", "Medium": "llama3.2:3b", "Large": "llama3.1:8b"}


def solvable_prompts(labels_path: str) -> set[str]:
    """Tasks no tier solves carry no signal about verification quality."""
    keep = set()
    if not os.path.exists(labels_path):
        return keep
    with open(labels_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                if rec["capability_tier"] != "Unsolved":
                    keep.add(rec["prompt"])
    return keep


def generate(llama, prompt: str, max_tokens: int) -> str:
    llama.reset()
    llama.eval(llama.tokenize(prompt.encode("utf-8")))
    _, text = InferenceEngine.generate_tokens(llama, max_tokens, temperature=0.0)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entry-tier", default="Small")
    ap.add_argument("--verifier-tiers", default="Small,Medium")
    ap.add_argument("--per-benchmark", type=int, default=40)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    keep = solvable_prompts(LABELS)

    candidates = load_candidates("data", args.per_benchmark, 42)
    if keep:
        candidates = [c for c in candidates if c["prompt"] in keep]
    if args.limit:
        candidates = candidates[: args.limit]
    print(f"Measuring Critic on {len(candidates)} solvable tasks "
          f"(entry tier {args.entry_tier})", flush=True)

    # Stage 1: one Executor answer per task at the entry tier, plus ground truth.
    answers = []
    llama = _load_model(TIER_MODELS[args.entry_tier], n_ctx=4096)
    try:
        for i, cand in enumerate(candidates, 1):
            bench = cand["benchmark"]
            budget = BENCHMARK_MAX_TOKENS.get(bench, 256)
            start = time.time()
            try:
                text = generate(llama, cand["prompt"], budget)
            except Exception as exc:
                print(f"  [{i}/{len(candidates)}] exec error: {exc}", flush=True)
                text = ""
            correct, score = is_correct(bench, text, cand["item"])
            answers.append({"cand": cand, "answer": text, "correct": correct, "score": score})
            print(f"  [{i}/{len(candidates)}] {bench:9} exec "
                  f"{'CORRECT' if correct else 'wrong'} {time.time()-start:.1f}s", flush=True)
    finally:
        try:
            llama.close()
        except Exception:
            pass
        del llama
        gc.collect()

    n_correct = sum(1 for a in answers if a["correct"])
    print(f"\nEntry-tier answers: {n_correct} correct / {len(answers)}", flush=True)

    # Stage 2: each verifier tier judges the same fixed set of answers.
    results = {}
    for tier in [t.strip() for t in args.verifier_tiers.split(",") if t.strip()]:
        print(f"\n=== Critic at {tier} ({TIER_MODELS[tier]}) ===", flush=True)
        llama = _load_model(TIER_MODELS[tier], n_ctx=4096)
        rows = []
        try:
            for i, a in enumerate(answers, 1):
                cand = a["cand"]
                bb = Blackboard(
                    task_id=str(i),
                    benchmark=cand["benchmark"],
                    prompt=cand["prompt"],
                )
                bb.post("Executor", "Critic", DRAFT, a["answer"])
                critic = CriticAgent()
                prompt = f"{critic.system_prompt(bb)}\n{critic.build_prompt(bb)}"
                try:
                    verdict = generate(llama, prompt, critic.max_tokens)
                except Exception as exc:
                    print(f"  [{i}] critic error: {exc}", flush=True)
                    verdict = ""
                critic.consume(verdict, bb)
                approved = bool(bb.latest(CRITIQUE).metadata.get("approved", True))
                rows.append({
                    "benchmark": cand["benchmark"],
                    "verifier_tier": tier,
                    "answer_correct": a["correct"],
                    "critic_approved": approved,
                    "verdict_text": verdict.strip()[:160],
                })
                with open(args.out, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rows[-1]) + "\n")
        finally:
            try:
                llama.close()
            except Exception:
                pass
            del llama
            gc.collect()
        results[tier] = rows

    # Stage 3: confusion matrix per verifier tier.
    print("\n=== Critic quality ===")
    hdr = "  {:<10} {:>6} {:>6} {:>6} {:>6} {:>12} {:>12}"
    print(hdr.format("verifier", "TP", "FN", "TN", "FP", "sensitivity", "specificity"))
    for tier, rows in results.items():
        tp = sum(1 for r in rows if not r["answer_correct"] and not r["critic_approved"])
        fn = sum(1 for r in rows if not r["answer_correct"] and r["critic_approved"])
        tn = sum(1 for r in rows if r["answer_correct"] and r["critic_approved"])
        fp = sum(1 for r in rows if r["answer_correct"] and not r["critic_approved"])
        sens = tp / (tp + fn) if (tp + fn) else float("nan")
        spec = tn / (tn + fp) if (tn + fp) else float("nan")
        print(hdr.format(tier, tp, fn, tn, fp, f"{sens:.3f}", f"{spec:.3f}"))

    print("\n  sensitivity = rejects a wrong answer (drives cascade solve rate)")
    print("  specificity = accepts a right answer (drives cascade cost)")
    always_approve = sum(1 for rows in results.values() for r in rows if r["critic_approved"])
    total = sum(len(rows) for rows in results.values())
    if total:
        print(f"\n  Critic approved {always_approve}/{total} answers overall. A Critic that "
              f"approves everything has sensitivity 0 and cannot drive a cascade.")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()

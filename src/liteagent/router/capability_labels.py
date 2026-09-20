"""
Capability-grounded labelling for the router validation set.

The original validation labels were assigned by a rubric defined in terms of
prompt length, which is also the router's dominant input feature. That made the
target circular: a single length threshold reproduces those labels better than
the router does, so the router was being measured against a restatement of one
of its own features rather than against task difficulty.

This module replaces the rubric with an observable definition used by the
routing literature (RouteLLM, Hybrid LLM): the correct tier for a task is the
**smallest model that actually solves it**. Labels are therefore measured, not
asserted, and no annotator opinion enters the target.
"""

import gc
import json
import os
import time

from liteagent.eval.metrics.gsm8k_metric import score_gsm8k
from liteagent.eval.metrics.hotpotqa_metric import score_hotpotqa
from liteagent.eval.metrics.humaneval_metric import score_humaneval
from liteagent.utils.inference import InferenceEngine

# Smallest-to-largest. A task is labelled by the first tier that solves it.
TIER_ORDER = [
    ("Low", "llama3.2:1b"),
    ("Medium", "llama3.2:3b"),
    ("High", "llama3.1:8b"),
]

# HotpotQA is scored by span overlap, so "solved" needs a threshold rather than
# exact match; 0.6 F1 is the conventional cut for treating a span as correct.
HOTPOTQA_F1_THRESHOLD = 0.6


def format_hotpotqa_prompt(item: dict) -> str:
    paragraphs = []
    for title, sentences in item.get("context", []):
        paragraphs.append(f"[{title}]: {''.join(sentences)}")
    return (
        "Context:\n" + "\n".join(paragraphs) +
        f"\n\nQuestion: {item.get('question', '')}\nAnswer in a short span:"
    )


def build_prompt(benchmark: str, item: dict, few_shot: bool = True) -> str:
    """
    Builds the prompt exactly as the harness would.

    The labels must be produced under the same protocol the system is evaluated
    under: measured zero-shot, llama3.2:1b reasons correctly but omits the
    answer marker, so extraction scores correct solves as failures and the
    resulting labels understate every tier's capability.
    """
    from liteagent.eval.fewshot import build_prefix
    from liteagent.eval.harness import BENCHMARK_SYSTEM_PROMPTS

    if benchmark == "gsm8k":
        body = item.get("question", "")
    elif benchmark == "hotpotqa":
        body = format_hotpotqa_prompt(item)
    elif benchmark == "humaneval":
        body = item.get("prompt", "")
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")

    system = BENCHMARK_SYSTEM_PROMPTS.get(benchmark, "")
    prefix = build_prefix(benchmark, few_shot)
    return f"{system}\n{prefix}{body}" if system else f"{prefix}{body}"


def is_correct(benchmark: str, response: str, item: dict) -> tuple[bool, float]:
    """Returns (solved, raw_score) for one model response."""
    if benchmark == "gsm8k":
        score = score_gsm8k(response, item.get("answer", ""))
        return bool(score >= 1.0), float(score)
    if benchmark == "hotpotqa":
        scores = score_hotpotqa(response, item.get("answer", ""))
        return bool(scores["f1"] >= HOTPOTQA_F1_THRESHOLD), float(scores["f1"])
    if benchmark == "humaneval":
        res = score_humaneval(response, item.get("test", ""), timeout=5.0)
        return bool(res["pass_status"] >= 1.0), float(res["pass_status"])
    raise ValueError(f"Unknown benchmark: {benchmark}")


def load_candidates(data_dir: str = "data", per_benchmark: int = 40, seed: int = 42) -> list[dict]:
    """
    Draws candidate prompts from the real benchmark subsets.

    Using benchmark items rather than hand-written prompts removes the
    "author invented both the prompts and their labels" objection.
    """
    import random
    rng = random.Random(seed)
    candidates = []

    gsm = [json.loads(l) for l in open(os.path.join(data_dir, "gsm8k", "subset.jsonl"), encoding="utf-8") if l.strip()]
    human = [json.loads(l) for l in open(os.path.join(data_dir, "humaneval", "subset.jsonl"), encoding="utf-8") if l.strip()]
    hotpot = json.load(open(os.path.join(data_dir, "hotpotqa", "subset.json"), encoding="utf-8"))

    for benchmark, items in (("gsm8k", gsm), ("humaneval", human), ("hotpotqa", hotpot)):
        pool = list(items)
        rng.shuffle(pool)
        for item in pool[:per_benchmark]:
            candidates.append({
                "benchmark": benchmark,
                "item": item,
                "prompt": build_prompt(benchmark, item),
                # HotpotQA ships an independent human difficulty label from its
                # original authors - a genuine external annotator for that slice.
                "external_difficulty": item.get("level"),
            })
    return candidates


def _load_model(model_tag: str, n_ctx: int = 4096, n_gpu_layers: int = None):
    import liteagent  # noqa: F401  - registers the CUDA DLL search path
    from llama_cpp import Llama
    from liteagent.config import WORKSTATION_N_GPU_LAYERS
    from liteagent.utils.model_resolver import resolve_model_path

    # Labelling runs on the workstation, so offload by default.
    if n_gpu_layers is None:
        n_gpu_layers = WORKSTATION_N_GPU_LAYERS
    return Llama(
        model_path=resolve_model_path(model_tag),
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        verbose=False,
        seed=42,
    )


def run_capability_pass(
    candidates: list[dict],
    output_path: str,
    max_tokens: int = 320,
    n_ctx: int = 4096,
    progress: bool = True,
) -> list[dict]:
    """
    Staged evaluation: each model is loaded once and run only on the tasks that
    smaller models failed, so the cost is far below (models x tasks).

    Results are appended to `output_path` as they are produced, so a long CPU
    run can be resumed rather than restarted.
    """
    pending = list(range(len(candidates)))
    labels: dict[int, dict] = {}

    for tier_name, model_tag in TIER_ORDER:
        if not pending:
            break
        if progress:
            print(f"\n=== Tier {tier_name} ({model_tag}): {len(pending)} task(s) to attempt ===", flush=True)

        llama = _load_model(model_tag, n_ctx=n_ctx)
        still_pending = []
        try:
            for n, idx in enumerate(pending, 1):
                cand = candidates[idx]
                start = time.time()
                # Match the harness budget per benchmark: a label produced under
                # a different token budget does not transfer to evaluation.
                from liteagent.eval.harness import BENCHMARK_MAX_TOKENS
                budget = BENCHMARK_MAX_TOKENS.get(cand["benchmark"], max_tokens)
                try:
                    # Decode through the same path the system under test uses, so
                    # the labels reflect how LiteAgent actually generates rather
                    # than how llama.cpp's own sampler would.
                    llama.reset()
                    llama.eval(llama.tokenize(cand["prompt"].encode("utf-8")))
                    _, response = InferenceEngine.generate_tokens(
                        llama, budget, temperature=0.0
                    )
                except Exception as exc:
                    response = ""
                    if progress:
                        print(f"  [{n}/{len(pending)}] inference error: {exc}", flush=True)

                solved, raw = is_correct(cand["benchmark"], response, cand["item"])
                elapsed = time.time() - start

                if solved:
                    record = {
                        "benchmark": cand["benchmark"],
                        "prompt": cand["prompt"],
                        "capability_tier": tier_name,
                        "solved_by": model_tag,
                        "raw_score": raw,
                        "external_difficulty": cand["external_difficulty"],
                        "latency_s": round(elapsed, 2),
                    }
                    labels[idx] = record
                    with open(output_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(record) + "\n")
                else:
                    still_pending.append(idx)

                if progress:
                    print(f"  [{n}/{len(pending)}] {cand['benchmark']:9} "
                          f"{'SOLVED' if solved else 'failed'} score={raw:.2f} {elapsed:.1f}s", flush=True)
        finally:
            # Each instance holds the weights plus an (n_ctx x n_vocab) float32
            # score buffer (~2 GB at n_ctx=4096). Relying on refcount timing to
            # release three of those in sequence is what exhausted memory and
            # killed an earlier run partway through the 8B tier.
            try:
                llama.close()
            except Exception:
                pass
            del llama
            gc.collect()

        pending = still_pending

    # Anything no tier solved is genuinely beyond the model pool; it carries no
    # routing signal (no tier is "correct"), so it is recorded and excluded.
    for idx in pending:
        cand = candidates[idx]
        record = {
            "benchmark": cand["benchmark"],
            "prompt": cand["prompt"],
            "capability_tier": "Unsolved",
            "solved_by": None,
            "raw_score": 0.0,
            "external_difficulty": cand["external_difficulty"],
        }
        labels[idx] = record
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    return [labels[i] for i in sorted(labels)]


def cohens_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    """Cohen's kappa between two label sequences over the same items."""
    if not labels_a or len(labels_a) != len(labels_b):
        return 0.0
    categories = sorted(set(labels_a) | set(labels_b))
    n = len(labels_a)
    observed = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n
    expected = sum(
        (labels_a.count(c) / n) * (labels_b.count(c) / n) for c in categories
    )
    if expected >= 1.0:
        return 1.0
    return (observed - expected) / (1.0 - expected)

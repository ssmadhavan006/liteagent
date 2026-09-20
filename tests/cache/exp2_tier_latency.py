"""
Experiment 2 (H2): three-tier restoration latency and losslessness.

The prefix-cache change showed that caching helps, but only ever exercised the
HOT tier. H2 claims a *three-tier* hierarchy, which requires showing that
Standby (RAM) and Cold (SSD) restores are also faster than recomputation. Those
tiers only engage under memory pressure, so this experiment creates that
pressure deliberately rather than hoping it appears.

Method
------
`max_ram_states` is set below the number of live contexts, so saving pushes
older entries out of Standby onto SSD. Each tier is then driven on comparable
content:

    MISS      a context never cached - full recomputation, the baseline
    COLD      evicted to SSD, restored from disk
    STANDBY   resident in host RAM, not the active slot
    HOT       the active slot

Each measurement reports restoration time and time-to-first-token, TTFT being
the quantity a restored context is meant to reduce.

Losslessness is checked separately: greedy decoding from a restored context must
emit byte-identical output to greedy decoding after a fresh prefill. A cache that
is fast but changes the answer is not a cache.

Usage:
    uv run python -m tests.cache.exp2_tier_latency --contexts 6 --ram-slots 2
"""

import argparse
import gc
import hashlib
import json
import os
import statistics
import time

from liteagent.cache import KVCacheManager
from liteagent.utils.inference import InferenceEngine

MODEL_TAGS = {"Small": "llama3.2:1b", "Medium": "llama3.2:3b", "Large": "llama3.1:8b"}
ROLES = ["Critic", "Executor", "Planner", "Retriever"]
OUT = "experiments/exp2_tier_latency.jsonl"


def make_context(i: int, target_tokens: int) -> str:
    """A distinct, realistically long context per slot."""
    filler = (
        "The agent maintains a running transcript of its reasoning so that later "
        "turns can refer back to earlier conclusions without restating them. "
    )
    repeats = max(1, target_tokens // 20)
    return f"Session {i} transcript.\n" + (filler * repeats) + f"\nEnd of session {i}.\n"


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_model(tier: str, n_ctx: int):
    import liteagent  # noqa: F401
    from llama_cpp import Llama
    from liteagent.config import EDGE_N_GPU_LAYERS
    from liteagent.utils.model_resolver import resolve_model_path
    return Llama(model_path=resolve_model_path(MODEL_TAGS[tier]), n_ctx=n_ctx,
                 n_gpu_layers=EDGE_N_GPU_LAYERS, verbose=False, seed=42)


def prefill(llama, text: str) -> int:
    llama.reset()
    tokens = llama.tokenize(text.encode("utf-8"))
    llama.eval(tokens)
    return len(tokens)


def first_token_ms(llama) -> float:
    start = time.perf_counter()
    InferenceEngine.sample_next_token(llama, temperature=0.0)
    return (time.perf_counter() - start) * 1000.0


def measure(mgr, llama, key, text, model_tag, n_ctx, expect=None) -> dict:
    """Restores one context and times it end to end."""
    t0 = time.perf_counter()
    tier = mgr.load_cache(session_key=key, llama_instance=llama, model_tag=model_tag,
                          ctx_size=n_ctx, prompt_hash=_hash(text))
    restore_ms = (time.perf_counter() - t0) * 1000.0

    recompute_ms = 0.0
    tokens = 0
    if tier == "MISS":
        t1 = time.perf_counter()
        tokens = prefill(llama, text)
        recompute_ms = (time.perf_counter() - t1) * 1000.0

    ttft = restore_ms + recompute_ms + first_token_ms(llama)
    if expect and tier != expect:
        print(f"    (expected {expect}, observed {tier})", flush=True)
    return {"tier": tier, "restore_ms": round(restore_ms, 2),
            "recompute_ms": round(recompute_ms, 2), "ttft_ms": round(ttft, 2),
            "recomputed_tokens": tokens}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="Small", choices=list(MODEL_TAGS))
    ap.add_argument("--contexts", type=int, default=6)
    ap.add_argument("--ram-slots", type=int, default=2)
    ap.add_argument("--context-tokens", type=int, default=400)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--n-ctx", type=int, default=4096)
    ap.add_argument("--ssd-dir", default="experiments/exp2_ssd")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if os.path.isdir(args.ssd_dir):
        import shutil
        shutil.rmtree(args.ssd_dir)

    texts = [make_context(i, args.context_tokens) for i in range(args.contexts)]
    mgr = KVCacheManager(max_ram_states=args.ram_slots, ssd_dir=args.ssd_dir,
                         log_dir="experiments")
    llama = load_model(args.tier, args.n_ctx)
    model_tag = MODEL_TAGS[args.tier]
    records = []

    try:
        # Populate. With contexts > ram_slots, early entries are evicted to SSD.
        print(f"Populating {args.contexts} contexts into {args.ram_slots} RAM slots "
              f"({model_tag})", flush=True)
        for i, text in enumerate(texts):
            n = prefill(llama, text)
            mgr.save_cache(session_key=f"ctx{i}", agent_role=ROLES[i % len(ROLES)],
                           llama_instance=llama, model_tag=model_tag,
                           ctx_size=args.n_ctx, prompt_hash=_hash(text))
            if i == 0:
                print(f"  context size: {n} tokens", flush=True)

        standby = set(mgr.storage.standby_cache.keys())
        cold = {f"ctx{i}" for i in range(args.contexts)} - standby
        print(f"  standby: {sorted(standby)}   cold(SSD): {sorted(cold)}\n", flush=True)

        if not cold:
            print("  WARNING: nothing was evicted; raise --contexts or lower --ram-slots")

        for r in range(args.repeats):
            # MISS: a context the cache has never seen.
            fresh = make_context(1000 + r, args.context_tokens)
            rec = measure(mgr, llama, f"unseen{r}", fresh, model_tag, args.n_ctx, "MISS")
            records.append(rec | {"repeat": r, "label": "MISS (recompute)"})
            print(f"  [{r}] MISS     ttft={rec['ttft_ms']:8.1f}ms  "
                  f"recompute={rec['recompute_ms']:.1f}ms ({rec['recomputed_tokens']} tok)", flush=True)

            # COLD: evicted to SSD. Restoring promotes it back to Standby.
            if cold:
                key = sorted(cold)[0]
                idx = int(key[3:])
                rec = measure(mgr, llama, key, texts[idx], model_tag, args.n_ctx, "COLD")
                records.append(rec | {"repeat": r, "label": "COLD (SSD)"})
                print(f"  [{r}] COLD     ttft={rec['ttft_ms']:8.1f}ms  "
                      f"restore={rec['restore_ms']:.1f}ms", flush=True)

            # HOT: the slot just restored is now active.
            if cold:
                key = sorted(cold)[0]
                idx = int(key[3:])
                rec = measure(mgr, llama, key, texts[idx], model_tag, args.n_ctx, "HOT")
                records.append(rec | {"repeat": r, "label": "HOT (active)"})
                print(f"  [{r}] HOT      ttft={rec['ttft_ms']:8.1f}ms  "
                      f"restore={rec['restore_ms']:.1f}ms", flush=True)

            # STANDBY: resident in RAM but not the active slot. Touch another
            # context first so the target is no longer hot.
            resident = [k for k in mgr.storage.standby_cache.keys()]
            if len(resident) >= 2:
                other, target = resident[0], resident[1]
                measure(mgr, llama, other, texts[int(other[3:])], model_tag, args.n_ctx)
                rec = measure(mgr, llama, target, texts[int(target[3:])],
                              model_tag, args.n_ctx, "STANDBY")
                records.append(rec | {"repeat": r, "label": "STANDBY (RAM)"})
                print(f"  [{r}] STANDBY  ttft={rec['ttft_ms']:8.1f}ms  "
                      f"restore={rec['restore_ms']:.1f}ms", flush=True)

            # Re-evict so the next repeat starts from the same shape.
            for i, text in enumerate(texts):
                prefill(llama, text)
                mgr.save_cache(session_key=f"ctx{i}", agent_role=ROLES[i % len(ROLES)],
                               llama_instance=llama, model_tag=model_tag,
                               ctx_size=args.n_ctx, prompt_hash=_hash(text))
            standby = set(mgr.storage.standby_cache.keys())
            cold = {f"ctx{i}" for i in range(args.contexts)} - standby

        # Losslessness: greedy decode from a restored context must match a fresh
        # prefill byte for byte.
        print("\n=== Losslessness ===", flush=True)
        text = texts[0]
        prefill(llama, text)
        fresh_tokens, fresh_text = InferenceEngine.generate_tokens(llama, 24, temperature=0.0)
        # Generation advanced the context past the saved state. Declaring that is
        # the caller's contract; without it the manager would serve a stale HOT.
        mgr.mark_context_dirty()

        verdicts = []
        for _ in range(3):
            tier = mgr.load_cache(session_key="ctx0", llama_instance=llama,
                                  model_tag=model_tag, ctx_size=args.n_ctx,
                                  prompt_hash=_hash(text))
            toks, txt = InferenceEngine.generate_tokens(llama, 24, temperature=0.0)
            mgr.mark_context_dirty()
            ok = (toks == fresh_tokens) and (txt == fresh_text)
            verdicts.append({"observed_tier": tier, "identical": ok})
            print(f"  restored from {tier:8} -> {'IDENTICAL' if ok else 'DIVERGED'}", flush=True)
            if not ok:
                print(f"    fresh:    {fresh_text[:80]!r}")
                print(f"    restored: {txt[:80]!r}")

        stale_hot = [v for v in verdicts if v["observed_tier"] == "HOT"]
        if stale_hot:
            print("  ERROR: HOT served after the context was mutated; the dirty flag "
                  "is not being honoured.", flush=True)

        with open(args.out, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
            f.write(json.dumps({"losslessness": verdicts}) + "\n")

        # Summary.
        print("\n=== TTFT by tier ===")
        print(f"  {'tier':<18}{'n':>3}{'median TTFT':>14}{'median restore':>16}{'speedup vs MISS':>18}")
        by = {}
        for rec in records:
            by.setdefault(rec["label"], []).append(rec)
        miss_med = statistics.median([r["ttft_ms"] for r in by.get("MISS (recompute)", [])] or [0])
        for label in ("MISS (recompute)", "COLD (SSD)", "STANDBY (RAM)", "HOT (active)"):
            rs = by.get(label)
            if not rs:
                continue
            med = statistics.median([r["ttft_ms"] for r in rs])
            rest = statistics.median([r["restore_ms"] for r in rs])
            sp = f"{miss_med/med:.1f}x" if med else "-"
            print(f"  {label:<18}{len(rs):>3}{med:>13.1f}ms{rest:>15.1f}ms{sp:>18}")
        print(f"\nWrote {args.out}")
    finally:
        try:
            llama.close()
        except Exception:
            pass
        del llama
        gc.collect()


if __name__ == "__main__":
    main()

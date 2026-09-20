# Experiment 2 — Three-Tier Cache Restoration (H2)

**Date:** 2026-09-20
**Status:** H2 supported. First measured evidence for the cache hierarchy.

Reproduce: `uv run python -m tests.cache.exp2_tier_latency --contexts 6 --ram-slots 2`

---

## 1. Why this had to be forced

Before this experiment the cache had never served a restore. Lifetime counters
across the project were **112 `LOAD_MISS` against 2 `LOAD_HIT`**, with zero
restores from Standby or Cold despite 35 evictions to SSD. States were written
and never read back, so nothing in the project had ever exercised the mechanism
H2 is about.

The cause was structural: entries were keyed on `sha256(full_prompt)` under a
session key unique per task. Every benchmark task carries a distinct prompt, so
every lookup was a guaranteed miss.

Two changes were needed before H2 could be measured at all:

1. **Prefix caching.** The reusable span is the output contract plus the
   few-shot exemplars, byte-identical across every task in a benchmark. It is
   evaluated once, snapshotted, and restored thereafter.
2. **Deliberate memory pressure.** Standby and Cold only engage when live
   contexts exceed `max_ram_states`. This experiment sets 6 contexts against 2
   RAM slots so eviction actually happens, rather than assuming it will.

## 2. Setup

- Model `llama3.2:1b`, CPU, `n_ctx=4096`, greedy decoding, seed 42.
- 6 distinct contexts of ~493 tokens each, 2 Standby slots, so 4 are evicted to SSD.
- 3 repeats; each tier driven on comparable content.
- TTFT = restoration + any recomputation + first sampled token.

## 3. Restoration latency by tier

| Tier | n | Median TTFT | Median restore | Speedup vs recompute |
| :--- | :---: | ---: | ---: | ---: |
| MISS (full recompute, 495 tok) | 3 | 2077.4 ms | — | 1.0x |
| COLD (SSD) | 3 | 320.8 ms | 320.7 ms | **6.5x** |
| STANDBY (host RAM) | 3 | 106.8 ms | 106.6 ms | **19.5x** |
| HOT (already resident) | 3 | 0.2 ms | 0.1 ms | (see §5) |

The hierarchy is monotonic and every tier beats recomputation, which is what H2
predicts. Restoration dominates TTFT in all cached cases: the cost is
deserialising state, not generating.

**End-to-end confirmation.** Running the real harness over GSM8K with prefix
caching enabled reproduces the effect on the actual pipeline: the first task
misses (482 prefill tokens, TTFT 2319 ms) and subsequent tasks are served from
Standby (46–105 prefill tokens, TTFT 195–444 ms), a ~9x reduction with 424 of
~480 prefix tokens restored rather than recomputed.

## 4. Losslessness

Greedy decoding (temperature 0, fixed seed) from a restored context must emit
byte-identical output to decoding after a fresh prefill.

| Restored from | Identical to fresh prefill |
| :--- | :---: |
| COLD (SSD) | yes |
| STANDBY (RAM) | yes |
| STANDBY (RAM) | yes |

## 5. A correctness defect this experiment caught

The losslessness check initially failed on HOT, and the cause was real rather
than a harness artefact.

`load_cache` returns `"HOT"` **without deserialising**: it assumes the model
still holds the context that was saved. Generation breaks that assumption — the
context has advanced by the generated tokens. With prefix caching in place, each
task was therefore being evaluated on top of the *previous* task's context while
the system reported a cache hit.

Fixed by tracking whether the resident context is still clean. Callers declare a
mutation via `mark_context_dirty()`, and a dirty HOT falls through to Standby for
a real restore. `tests/cache/test_hot_invalidation.py` pins the invariant.

**Consequence for reporting: HOT is effectively unreachable in the pipeline**,
because every task generates and so dirties the context. The honest steady-state
figure is the Standby one, **~19.5x**, not the 12000x HOT number. HOT is reported
here only as the floor on restoration cost.

## 6. Threats to validity

- **Single model, single machine.** 1B on CPU. Restore cost scales with state
  size, so larger models will show different absolute numbers; the ordering
  should hold but has not been measured.
- **Cold-tier timing reflects a warm OS page cache.** States were written
  moments earlier, so 320 ms is a best case for SSD. A cold page cache, and the
  Pi's slower storage, will be worse — `hardware_inventory.md` still records the
  Pi's storage speed as unverified.
- **Synthetic contexts.** §3's contexts are generated filler of uniform size.
  The end-to-end GSM8K figures are real prompts and agree in direction.
- **Prefix reuse is the mechanism under test**, not multi-agent context sharing.
  Agents still build independent prompts; they share only the exemplar prefix.
- **Prompt restructuring was A/B tested** on identical tasks: the prefix-first
  construction scored 3/8 against 4/8 for the original, a one-task difference at
  n=8 that this sample cannot resolve. No accuracy claim either way.

## 7. Status of H2

> **H2**: A persistent three-tier KV cache reduces context restoration latency
> compared with flat in-memory caching.

Supported for the restoration-latency half: all three tiers beat recomputation,
the ordering is monotonic, and restoration is lossless. The comparison against
the flat in-memory baseline (`flat_cache`) still has to be run in Phase 8 — this
experiment establishes the hierarchy against recomputation, not against a
competing cache design.

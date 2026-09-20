# Phase 8 — First Valid Results (GSM8K, Medium tier)

**Date:** 2026-09-20
**Scope:** 25 GSM8K tasks, `llama3.2:3b`, CPU, few-shot protocol, greedy decoding.

Reproduce:

```
uv run python -m liteagent.eval.run_evaluation --config always_medium      --dataset gsm8k --limit 25
uv run python -m liteagent.eval.run_evaluation --config flat_cache_medium  --dataset gsm8k --limit 25
```

> All earlier result records (90 rows, `schema_version: 1`) are void — produced
> with the broken decoder, the non-executing HumanEval scorer, zero-shot
> prompting and invalid energy. Archived as `VOID_schema_v1_results.bak`. These
> are the project's first results that measure the system rather than the harness.

---

## 1. Results

| Config | n | Quality | Median TTFT | Median latency | Agent turns run |
| :--- | :---: | :---: | ---: | ---: | :--- |
| `always_medium` (three-tier cache, with Critic) | 25 | 0.480 | 1568 ms | 72.8 s | Planner 25, Executor 34, Critic 34 |
| `medium_no_critic` (three-tier cache) | 25 | **0.640** | **523 ms** | **16.5 s** | Planner 25, Executor 25 |
| `flat_cache_medium` (single-tier cache) | 25 | 0.640 | 555 ms | 20.5 s | Planner 25, Executor 25, **Critic 0 (all failed)** |

Cache behaviour across agent turns: the three-tier configuration served 90
Standby restores against 3 misses.

## 2. The Critic degrades output and must not be in the default chain

The Critic was run 34 times (25 tasks plus 9 revisions) and made results worse.

| Comparison | Result |
| :--- | :--- |
| Accuracy with vs without Critic | 0.480 vs **0.640** |
| Median latency | 72.8 s vs **16.5 s** (4.4x) |
| Paired over 25 identical tasks | removing it **won 4, lost 0, tied 21** |

It did not improve a single task. Both arms use the same cache, tier, protocol
and prompts, so the cache cannot account for the difference — restores are
byte-identical (Exp 2 §4).

This is the predicted consequence of the H1b measurement: the Critic's Youden's
J is −0.098 at 1B and +0.092 at 3B, meaning its verdicts are near-independent of
whether an answer is correct. Revisions triggered on that basis are close to
random edits applied to answers that were already right. H1b showed the verifier
could not discriminate; this shows what that costs.

**Action taken:** the Critic is removed from the default active set in
`router/pruning.py`. It remains implemented and is still used by the `cascade`
configuration, where a rejection drives escalation rather than revision.

*Caveat:* 4 wins against 0 losses among 4 discordant pairs is a one-sided sign
test at p ≈ 0.06. The direction is unambiguous and the latency saving is large
and certain, but the accuracy effect alone is not significant at n = 25.

## 3. What the cache comparison does and does not show

The headline three-tier vs flat comparison **is not valid as a latency result**,
and the reason matters more than the numbers.

The flat cache holds 2 in-memory slots. The chain needs 3 role prefixes
(Planner, Executor, Critic). Saving the third raised
`EXPECTED_LIMIT_REACHED: Max in-memory prefix slots exhausted` on **all 25
tasks**, so the Critic step failed and was skipped every time. The flat arm
therefore ran a two-agent chain. Its lower latency reflects doing less work, and
its higher accuracy reflects not running the harmful Critic — neither is a
property of flat caching.

Two things can be said honestly:

**(a) The flat cache cannot run this workload.** It fails 100% of Critic turns
under a chain that the tiered hierarchy completes by spilling to SSD. That is a
capacity result supporting H2, and it is what the `EXPECTED_LIMIT_REACHED`
failure mode was built to detect.

**(b) Like-for-like, the tiered cache is modestly faster.** Comparing the two
two-agent configurations — `medium_no_critic` (tiered) against
`flat_cache_medium` — gives 16.5 s vs 20.5 s median latency and 523 ms vs 555 ms
TTFT. Treat this as an *upper bound* on the tiered advantage: the flat arm also
paid for 25 failed Critic save attempts. A clean rerun with the flat cache
explicitly configured for a two-agent chain is still owed.

## 4. Threats to validity

- **n = 25, one benchmark, one tier.** GSM8K only, `llama3.2:3b` only. Nothing
  here exercises the 8B model: without a workstation gRPC server, Large-tier
  tasks fall back to local Medium, so `static_full`, `cache_only` and the
  high-complexity routes of `liteagent` have not been measured.
- **Energy is unmeasured**, recorded as `null`. The GPU backend correctly
  refuses to report while inference runs on CPU.
- **The flat-cache slot count (2) is a configuration choice**, not a property of
  flat caching. A flat cache with 4 slots would not have exhausted. The finding
  is that a *fixed-capacity* cache fails when concurrent contexts exceed it,
  where a tiered one degrades to disk.
- **Quality figures depend on the few-shot protocol**, fixed identically across
  configurations by the harness.

## 5. Status of the hypotheses

- **H1 (routing):** unchanged and negative. See `router_capability_analysis.md`.
- **H2 (cache):** supported on capacity (flat cache cannot run the workload) and
  on restoration latency against recomputation (Exp 2: 6.5x cold, 19.5x
  standby). The like-for-like latency advantage over flat caching is real but
  modest and needs the clean rerun in §3(b).
- **H3 (co-design):** still untestable while the routing half is negative.
- **H4 (edge):** blocked on hardware.

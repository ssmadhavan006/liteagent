# Phase 9 — Router Validation Against Capability-Grounded Labels

**Date:** 2026-09-20
**Status:** Negative result for H1's routing mechanism. Reported as measured.

---

## 1. Why the labels were rebuilt

The original 80-prompt validation set was labelled with a rubric defined in
terms of prompt length ("under 150 characters" = Low, "exceeding 800" = High).
Prompt length is also the complexity scorer's dominant feature, so the target
was a restatement of one of the model's own inputs.

The circularity is measurable. Fitting a *single length threshold* on the
original train split and evaluating on its held-out split:

| Predictor | Held-out accuracy (original labels) |
| :--- | :---: |
| `len(prompt)` thresholds, fit on train | **86.67%** (26/30) |
| LiteAgent 7-feature scorer | 56.67% (17/30) |

The router was 30 points worse than one of its own features, because the labels
encoded length rather than difficulty.

## 2. The replacement labelling scheme

Labels are now **measured, not asserted**: the correct tier for a task is the
smallest model that actually solves it, which is how the routing literature
(RouteLLM, Hybrid LLM) defines routing ground truth.

- 120 candidates drawn from the real GSM8K / HumanEval / HotpotQA subsets, so
  no prompt is author-invented.
- Each is attempted by `llama3.2:1b`, then `llama3.2:3b`, then `llama3.1:8b`,
  under the **same protocol used for evaluation** (few-shot prefix, output
  contract, per-benchmark token budget, same decode path).
- A task no tier solves carries no routing signal — no tier is "correct" — and
  is excluded.

Producer: `src/liteagent/router/capability_labels.py`.

### Distribution (n = 120)

| Tier (smallest model that solves it) | Count |
| :--- | :---: |
| Low (`llama3.2:1b`) | 35 |
| Medium (`llama3.2:3b`) | 42 |
| High (`llama3.1:8b`) | 17 |
| Unsolved by any tier | 26 |

**Usable labelled set: 94.**

## 3. Result: surface features do not predict capability

Stratified 61 train / 33 test split, seed 42. The scorer's exact functional form
(one linear score, two cutpoints) was refit by ordinal logistic regression.

| Predictor | Held-out accuracy | 95% CI |
| :--- | :---: | :---: |
| Majority class ("always Medium") | **45.45%** | 29.8–62.0% |
| Refit ordinal router | 45.45% | 29.8–62.0% |
| Shipped hand-tuned config | **27.27%** | 15.1–44.2% |

The refit model only reaches majority-class accuracy by *collapsing onto the
majority class* — under strong regularisation it predicts "Medium" for all 33
test items and performs no routing at all.

Weakening the regularisation shows the failure is not an optimisation artefact:

| L2 | Train | Test |
| :--- | :---: | :---: |
| 0.001 | 65.6% | 36.4% |
| 0.01 | 67.2% | 36.4% |
| 0.1 | 63.9% | 39.4% |
| 1.0 | 44.3% | 45.5% (all-Medium) |
| 10.0 | 44.3% | 45.5% (all-Medium) |

The features fit the training split to 67% and then generalise *below* the
majority baseline. That is the signature of fitting noise.

**Conclusion.** With this feature set and this sample size, prompt-surface
statistics carry no generalisable signal about which model tier can solve a
task. The shipped hand-tuned configuration is worse than a constant predictor.

### Corroboration from an independent annotator

HotpotQA ships a human difficulty label from its own authors. All 21 labelled
HotpotQA items in our set are annotated `hard` — yet `llama3.2:1b` solved 9 of
them. Task difficulty as a human judges it does not correspond to difficulty as
a model experiences it. This supports the interpretation that the problem is the
*target concept*, not merely our feature engineering.

## 4. The finding that does hold: tier value is concentrated in code

Computed over the full labelled set, the fraction of tasks that **require** the
Large tier (unsolved by both smaller models):

| Benchmark | Requires 8B | n |
| :--- | :---: | :---: |
| GSM8K | 6% (2) | 35 |
| HumanEval | **39% (15)** | 38 |
| HotpotQA | **0% (0)** | 21 |

The Large tier's value is almost entirely in code generation. On HotpotQA it
buys nothing: every task the 8B solves is already solved by a smaller model.

This is a routing-relevant result that does not depend on the complexity
scorer — but note the caveat in §5: a benchmark-only router still did not beat
majority class on this test split (36.36%, CI 22.2–53.4%), so the per-benchmark
rule is suggestive rather than established.

## 5. Threats to validity

- **Small sample.** 94 usable labels, 33 in the test split. The correct claim is
  "no signal detectable at this sample size", not "no signal exists". Confidence
  intervals span roughly 30 points and overlap for every predictor compared.
- **Narrow feature set.** Seven surface statistics. A learned semantic
  representation may well succeed where these fail; this result does not rule
  that out.
- **Capability labels are stochastic.** Greedy decoding makes a single attempt
  per tier deterministic, but a model that solves a task at one token budget or
  prompt format may fail at another. The label is "solved under this protocol".
- **Three-way labels from a 3-model pool.** Adding or changing a tier changes
  every label; these labels are not portable to a different model pool.

## 6. Consequences for the hypotheses

- **H1 (routing effectiveness)** cannot be supported through predicted quality.
  Routing may still reduce latency and energy, but any claim that it *maintains
  quality* must be measured directly rather than argued from router accuracy.
- **H2 (cache performance)** is unaffected. It does not depend on routing.
- **H3 (co-design synergy)** is weakened at its routing end. If the router
  cannot select tiers correctly, an interaction effect between routing and cache
  placement has much less room to appear.

No configuration change was made as a result of this analysis. The refit
parameters are not written to `config/router_config.yaml`, because the refit is
only competitive when it stops routing altogether — replacing a bad router with
a constant would remove the mechanism under study rather than improve it.

Reproduce with:

```
uv run python -m tests.router.calibrate_router
uv run python -m tests.router.analyze_labels
```

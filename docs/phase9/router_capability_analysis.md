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

## 7. Policy comparison and the resulting design change

Per-tier costs are the mean attempt latencies measured during labelling on this
hardware: Small 12.9 s, Medium 17.1 s, High 39.5 s.

| Policy | Solved | Cost |
| :--- | :---: | :---: |
| always-small | 37.2% | 1209 s |
| **always-medium** | **81.9%** | **1611 s** |
| oracle-router (unreachable) | 100% | 1841 s |
| shipped-router | 94.7% | 2751 s |
| cascade, perfect verifier | 100% | 2891 s |
| always-large (`static_full`) | 100% | 3711 s |

Three readings matter.

**The shipped router's solve rate overstates it.** It gets 94.7% of tasks onto a
sufficient tier while choosing the *correct* tier only 27% of the time, because
its errors skew toward over-provisioning. It buys quality with cost, not with
accuracy — and at 2751 s it is only 26% cheaper than never routing at all.

**always-medium is the baseline to beat.** 81.9% of solvable tasks at 43% of
always-large's cost. It was not previously in the comparator set, so no prior
result in this project was measured against it. It is now
(`always_small`/`always_medium`/`always_large` in the evaluation driver).

**A cascade is not free, but it dominates always-large.** Escalating pays for
every failed attempt, so even a perfect verifier costs 57% more than a perfect
router (2891 s vs 1841 s). It is still 22% cheaper than always-large at
identical 100% quality, and always-large is the current `static_full` baseline.

### Design change

Tier selection moves from prediction to observation. `AgentOrchestrator` gains
`escalate_on_reject`: a Critic rejection re-runs the task on the next tier up
instead of asking the same model again. The escalated attempt is issued fresh —
the feedback block is suppressed — so the larger model is not anchored to the
rejected answer, and the cache session key includes the tier because KV state is
model-specific.

This converts an unsolvable prediction problem into a measurable verification
problem: solve rate is bounded by the Critic's specificity, cost by its
sensitivity. **Both are currently unmeasured**, and they decide whether the
cascade is viable. Measuring Critic sensitivity/specificity against the
capability labels is the next experiment, and the cascade must not be claimed to
work until that number exists.

## 8. H1b — the Critic is not a usable verifier (measured)

The cascade's viability rests entirely on the Critic's ability to tell a correct
answer from a wrong one. That was measured directly: the entry tier
(`llama3.2:1b`) produced one answer for each of the 94 solvable tasks, ground
truth came from the benchmark metric, and the Critic then judged that same fixed
set of answers at two tiers.

| Verifier | n | TP | FN | TN | FP | Sensitivity | Specificity | Youden's J |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Critic @ 1B | 94 | 33 | 26 | 12 | 23 | 0.559 | 0.343 | **−0.098** |
| Critic @ 3B | 94 | 29 | 30 | 21 | 14 | 0.492 | 0.600 | **+0.092** |

Youden's J (`sensitivity + specificity − 1`) is 0 for a coin flip. The 1B Critic
scores **below** it; the 3B Critic is marginally above. The Critic's verdicts are
very nearly independent of whether the answer is actually correct.

Per benchmark, every cell but one is at or below chance:

| Verifier | GSM8K | HumanEval | HotpotQA |
| :--- | :---: | :---: | :---: |
| @ 1B | J = −0.20 | J = −0.06 | J = −0.28 |
| @ 3B | J = −0.02 | J = −0.01 | J = +0.25 |

The 1B/HumanEval cell is instructive: sensitivity 0.85, specificity 0.09. It is
not detecting wrong code — it is rejecting nearly everything, which scores well
on sensitivity alone. Reporting sensitivity without specificity would have made
that look like a success.

### Consequence: the cascade is Pareto-dominated

Substituting the measured verifier quality into the policy model:

| Policy | Solved | Cost |
| :--- | :---: | :---: |
| **always-medium** | **81.9%** | **1607 s** |
| cascade, measured Critic @ 3B | 55.9% | 2659 s |
| cascade, measured Critic @ 1B | 42.8% | 3586 s |
| always-large (`static_full`) | 100% | 3713 s |
| cascade, perfect verifier (hypothetical) | 100% | 2893 s |

With the Critic as built, the cascade is worse than a fixed mid-tier model on
**both** axes — lower solve rate and higher cost. Low specificity escalates
correct answers, so cost approaches always-large; low sensitivity lets wrong
answers through, so the solve rate collapses.

**H1b is refuted.** Verification-gated escalation cannot be claimed as a
contribution with this verifier.

## 9. Overall finding for H1

Both available mechanisms for tier selection fail on this system:

- **Prediction** (§3): 27.27% tier accuracy against a 45.45% constant predictor.
- **Verification** (§8): Youden's J of −0.098 (1B) and +0.092 (3B).

And a trivial policy dominates both: **always-medium solves 81.9% of solvable
tasks at 43% of always-large's cost**, beating the full routed system on cost
and the cascade on both axes.

The honest statement of H1 is therefore negative: *for this model pool and task
mix, no routing mechanism we implemented improves on fixed mid-tier selection.*
That is a result worth reporting, but it is not the result the project set out
to claim, and it should not be presented as one.

Scope limits, as in §5: 94 labelled tasks, one answer sampled per task under
greedy decoding, three specific models. A stronger verifier — self-consistency
across samples, a trained classifier over hidden states, or an execution-based
check for code — could change the conclusion, and none of those were tested.
HumanEval is the obvious candidate, since correctness there is decidable by
running the tests rather than by asking a model.

Reproduce with:

```
uv run python -m tests.router.calibrate_router
uv run python -m tests.router.analyze_labels
uv run python -m tests.router.cascade_analysis
uv run python -m tests.router.measure_critic
```

# LiteAgent — Novelty Matrix

This document isolates LiteAgent's contribution relative to prior LLM routing, KV-caching, and multi-agent systems.

> [!NOTE]
> Rows describe **mechanisms**, not deployment choices. Running on a particular
> board is a hardware decision, not a research contribution, so it is not a row
> here; edge operation appears only where it changes the mechanism (tier
> placement across heterogeneous devices).

## Capability Matrix

| Capability | RouteLLM | LMCache | KVFlow | Bao 2025 | Shkolnikov 2026 | Lu 2026 | LiteAgent (Ours) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Complexity-aware model routing** | ✓ | ✗ | ✗ | ✓ | ✗ | ✓ | ✓ |
| **Persistent KV cache across sessions** | ✗ | ✓ | ✓ | Partial | ✓ | ✓ | ✓ |
| **Multi-tier cache (VRAM/RAM/disk)** | ✗ | ✓ | Partial | ✗ | ✓ | ✓ | ✓ |
| **Agent-role-aware eviction** | ✗ | ✗ | ✓ | ✗ | Partial | ✗ | ✓ |
| **Multi-agent orchestration** | ✗ | ✗ | ✓ | ✗ | ✓ | ✗ | ✓ |
| **Routing/caching co-design** | ✗ | ✗ | ✗ | Partial | ✗ | ✓ | ✓ |
| **Heterogeneous physical devices** | ✗ | ✗ | ✗ | ✓ | ✗ | ✓ | ✓ |
| **Measured on real hardware** | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ (simulated) | ✓ (planned) |

Column sources: Bao et al. 2025 (arXiv 2508.11291), Shkolnikov 2026 (arXiv 2603.04428), Lu et al. 2026 (arXiv 2609.06940). See `annotated_bibliography.md` Cluster 5.

## Novelty Analysis

**No single row in this table is our contribution.** Every capability has prior
art, and the 2026-09 literature refresh showed the gap is narrower than the
original Phase 1 review assumed:

- Complexity routing: RouteLLM, Hybrid LLM, Bao et al.
- Persistent multi-tier KV cache: LMCache, Shkolnikov.
- Agent-aware cache scheduling: KVFlow.
- Multi-agent persistent cache on edge hardware: Shkolnikov.
- The routing/cache co-design thesis itself: Lu et al.

**The contribution is the last row combined with the rest.** Lu et al. (2026)
propose joint routing and cache management across device/edge/cloud, but
evaluate by workload-level analytical simulation and name system integration as
an open problem. Shkolnikov (2026) builds and measures multi-agent persistent
caching on edge hardware but does no routing. Bao et al. (2025) build and
measure complexity routing with cache-switching cost but for single-session
dialogue, not agent chains.

LiteAgent's claim is therefore **empirical, not conceptual**: the first built and
measured co-design of complexity routing with a persistent tiered KV hierarchy
for *multi-agent* workloads on *physically heterogeneous* devices. The mechanism
that makes it a co-design rather than an integration is that the router selects
the active agent set at runtime, so the cache hierarchy's workload is itself a
function of the routing decision, and role priority determines what survives
eviction under the resulting pressure.

**What would falsify our claim.** If Exp 3 (H3) shows the co-design effect is
merely additive — routing gains plus caching gains with no interaction — then
this is an engineering integration of known parts rather than a co-design, and we
will report it as such. Given how much of the surrounding space is now occupied,
a null result on H3 would mean this work belongs as a reproduction/measurement
study, not a novel-systems paper.

**What would falsify our claim.** If the co-design effect in Exp 3 (H3) proves
merely additive — routing gains plus caching gains, with no interaction — then
this reduces to an engineering integration of known parts, and we will report it
as such.

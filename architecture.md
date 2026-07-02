# LiteAgent — Architecture

## 1. System Overview
<TODO — filled in Phase 2>

## 2. Hardware Targets
<TODO — filled in Phase 0 — see docs/phase0/hardware_inventory.md, summarized here>

## 3. Model Tiers
<TODO — filled in Phase 0 — see docs/phase0/model_manifest.md, summarized here>

## 4. Agent Definitions
<TODO — filled in Phase 0 — see docs/phase0/system_contract.md>

## 5. Complexity Router
<TODO — filled in Phase 2/3>

## 6. Three-Tier KV-Cache Manager
<TODO — filled in Phase 2/4>

## 7. Edge–Workstation Communication (gRPC)
<TODO — filled in Phase 2/5>

## 8. Data Flow Diagram
<TODO — filled in Phase 2, as a text/mermaid diagram, not an image file>

## 9. Design Decisions Log
| Decision | Rationale | Phase | Date |
|---|---|---|---|
| Quantization format: GGUF Q4_K_M | Default quantization for edge deployment to fit small/medium models in local RAM. | 0 | 2026-07-02 |
| Environment and package manager: uv | Standard tool for reproducible environment management and speed. | 0 | 2026-07-02 |

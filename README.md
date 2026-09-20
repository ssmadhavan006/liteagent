# LiteAgent: Complexity-Aware Routing & KV-Cache Co-Design

LiteAgent is a research systems project that co-designs complexity-aware routing and a three-tier KV-cache manager for multi-agent LLM systems deployed on edge-workstation environments. By dynamically routing sub-tasks based on reasoning complexity and optimizing KV-cache management across local RAM, VRAM, and cold storage, LiteAgent aims to minimize end-to-end latency and resource overhead while maintaining benchmark-specific reasoning and execution accuracy.

---

## Canonical Documents

This repository relies on three root documents to govern development and trace progress:
1.  **[architecture.md](file:///d:/Coding/liteagent/architecture.md)**: The technical design, system overview, hardware targets, model tiers, and design decisions log.
2.  **`rules.md` (Local / Ignored)**: Core operating rules, reproducibility guidelines, and coding best practices.
3.  **`progress.md`**: The single source of truth for the project state and the reverse-chronological development checklist log. Tracked in repository for project state continuity.

---

## Repository Directory Map

```
liteagent/
├── docs/
│   └── phase0/
│       ├── system_contract.md      # I/O schemas, agent roles, metric definitions
│       ├── hardware_inventory.md   # Specs for local workstation and edge targets
│       └── model_manifest.md       # Model manifest, sizes, and pull commands
├── experiments/
│   └── README.md                   # CSV + Git logging convention
├── pyproject.toml                  # Python package and metadata configuration
├── .gitignore                      # Git exclusion rules
├── architecture.md                 # System architecture overview
└── README.md                       # Repository entry point (this file)
```

---

## Getting Started

### 1. Requirements
*   Python `>=3.12`
*   [uv](https://github.com/astral-sh/uv) (for environment and dependency management)
*   [Ollama](https://ollama.com/) (installed locally for model inference)

### 2. Pulling Models
Follow the instructions in [model_manifest.md](file:///d:/Coding/liteagent/docs/phase0/model_manifest.md) to download the three required model tiers (`llama3.2:1b`, `llama3.2:3b`, and `llama3.1:8b`).

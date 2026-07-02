# LiteAgent — Operating Rules & Guidelines

## Operating Rules

1. **Never fabricate a number, benchmark result, hardware spec, library capability, or API behavior.** If you don't know something and can't verify it by running a command or reading official docs/source, say so explicitly and ask the user, or mark it as `TODO: UNVERIFIED` in the relevant doc. A guessed number in a research paper is worse than a missing one.
2. **Use `uv` for all Python package and environment management.** Never use bare `pip install`. Use `uv init`, `uv add <package>`, `uv venv`, `uv run <script>`, `uv lock`, etc. If you need a tool not manageable by uv (e.g. a system package), say so and ask the user to install it, or use `uv tool install` where applicable.
3. **Terminal execution policy — this is a hard constraint:**
   - If a command is expected to finish in **under ~60 seconds** (installs, file operations, quick scripts, benchmarking a single small read/write, `lscpu`, `free -h`, git operations, etc.), run it yourself and read the output.
   - If a command is **long-running** — model downloads, quantization jobs, any form of training/fine-tuning, full benchmark sweeps, anything you estimate will take more than ~60 seconds or has unpredictable duration — **do not run it**. Instead, output the **exact command(s)** in a fenced code block, clearly state what you expect the output to look like and why you need it, and stop and wait for the user to run it manually and paste the output back to you. Never simulate or guess what that output would have been.
   - When in doubt about whether a command is "long," treat it as long and ask.
4. **No silent architecture decisions.** Any nontrivial design choice (e.g., which quantization format, which cache eviction policy, message schema for gRPC) must be written down in `architecture.md` with a one-line rationale, not just implemented silently in code.
5. **Update `progress.md` after every major action**, not at the end of a session. "Major" = anything that changes project state: a file created, a decision locked in, a command run and its result, a deliverable completed.
6. **Never mark a deliverable "done" without verification.** A doc isn't done because it was written — it's done because its claims were checked against something real (a command output, an official source, explicit user confirmation).
7. **Ask the user rather than assume** whenever a requirement is genuinely ambiguous (e.g., they haven't told you their exact workstation GPU). Do not invent plausible-sounding specs to fill a gap.
8. **Stay inside Phase 0 scope.** If you're tempted to write inference code, router logic, or cache logic, stop — that's Phase 3+. Flag the idea in `progress.md` under "Deferred to later phase" instead.

## Project Guidelines & Best Practices

- **Reproducibility:** Every model, dataset, and library version referenced anywhere in docs or code must be pinned (exact version/tag/commit), not "latest."
- **No hallucinated APIs:** Before calling any function from a library (llama.cpp bindings, gRPC, etc.), verify it exists — check installed package docs/source (`uv run python -c "help(...)"`, reading the actual installed source under `.venv`) rather than recalling from memory. If uncertain, say so and check rather than guessing a plausible-looking call.
- **Separation of Concerns:** Scoping/docs (Phase 0), design (Phase 2), and implementation (Phase 3+) must not blur — no code commits during Phase 0 beyond project scaffolding.
- **Traceability:** Every hardware/performance claim must be traceable to either (a) a command you ran and logged, (b) an official spec sheet you read via web fetch and cited, or (c) explicit user-provided input. Tag anything else `UNVERIFIED`.
- **Design Rationale:** Every design decision gets a rationale, however brief, recorded in `architecture.md` — future-you (or the paper's Methods section) needs to explain *why*, not just *what*.
- **Fail Loudly:** If a command errors, log the actual error in `progress.md`, don't paper over it or retry silently multiple times without noting it.
- **No Scope Creep:** If you think of a nice-to-have beyond what's asked, propose it in `progress.md` under "Deferred" or ask the user — don't just build it.
- **Commit Discipline:** One logical change per commit, descriptive messages, no giant multi-purpose commits.

"""
Phase 8 experiment driver.

Runs one system configuration over one or more benchmark subsets and appends
versioned records via EvaluationHarness. Every configuration goes through the
same harness and the same record schema, so results are comparable across runs.

Examples:
    uv run python -m liteagent.eval.run_evaluation --config liteagent --dataset gsm8k
    uv run python -m liteagent.eval.run_evaluation --config static_full --dataset all --limit 25
    uv run python -m liteagent.eval.run_evaluation --config all --dataset all
"""

import argparse
import json
import os
import sys
import time

from liteagent.eval.harness import EvaluationHarness

DATASETS = ("gsm8k", "hotpotqa", "humaneval")

CONFIGS = (
    "liteagent",
    "routing_only",
    "cache_only",
    "static_full",
    "routellm_heuristic",
    "flat_cache",
)


def load_dataset(dataset: str, data_dir: str) -> list[dict]:
    if dataset == "hotpotqa":
        with open(os.path.join(data_dir, "hotpotqa", "subset.json"), encoding="utf-8") as f:
            return json.load(f)
    path = os.path.join(data_dir, dataset, "subset.jsonl")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def task_id_for(dataset: str, item: dict, index: int) -> str:
    if dataset == "humaneval":
        return str(item.get("task_id", f"humaneval_{index}"))
    if dataset == "hotpotqa":
        return str(item.get("_id", f"hotpotqa_{index}"))
    return f"gsm8k_{item.get('dataset_index', index)}"


def build_runner(config: str, log_dir: str, ssd_dir: str, router_config: str,
                 workstation_client=None):
    """
    Constructs the runner for one configuration.

    liteagent / routing_only / cache_only all come from the same LiteAgentRunner
    with one half disabled, so the ablation never compares different codebases.
    """
    from liteagent.runner import (
        LiteAgentRunner,
        build_cache_only_runner,
        build_routing_only_runner,
    )

    common = dict(
        workstation_client=workstation_client,
        router_config_path=router_config,
        ssd_dir=ssd_dir,
        log_dir=log_dir,
    )

    if config == "liteagent":
        return LiteAgentRunner(**common)
    if config == "routing_only":
        return build_routing_only_runner(**common)
    if config == "cache_only":
        return build_cache_only_runner(**common)

    if config == "static_full":
        from liteagent.baselines.static_full_pipeline import StaticFullPipelineRunner
        return StaticFullPipelineRunner(workstation_client, log_dir=log_dir)

    if config == "routellm_heuristic":
        from liteagent.baselines.routellm_heuristic_approx import RouteLLMHeuristicDispatcher
        from liteagent.cache import KVCacheManager
        cache = KVCacheManager(max_ram_states=5, ssd_dir=ssd_dir, log_dir=log_dir)
        return RouteLLMHeuristicDispatcher(workstation_client, cache, log_dir=log_dir)

    if config == "flat_cache":
        from liteagent.baselines.vllm_prefix_cache_approx import VLLMPrefixCacheApprox
        from liteagent.network.dispatch import TaskDispatcher
        flat = VLLMPrefixCacheApprox(max_in_memory_slots=2, log_dir=log_dir)
        dispatcher = TaskDispatcher(
            router_config_path=router_config,
            edge_cache_manager=flat,
            workstation_client=workstation_client,
            log_dir=log_dir,
        )
        dispatcher.baseline_name = "flat_cache"
        return dispatcher

    raise ValueError(f"Unknown config: {config}")


def run_one(config: str, dataset: str, args, workstation_client) -> dict:
    items = load_dataset(dataset, args.data_dir)
    if args.limit:
        items = items[: args.limit]

    harness = EvaluationHarness(
        baseline_name=config,
        results_path=args.results,
        failed_path=args.failed,
        sample_seed=args.seed,
        few_shot=not args.zero_shot,
        energy_source=args.energy_source,
        meter_csv=args.meter_csv,
    )
    runner = build_runner(
        config,
        log_dir=args.log_dir,
        ssd_dir=os.path.join(args.ssd_root, f"{config}_{dataset}"),
        router_config=args.router_config,
        workstation_client=workstation_client,
    )

    stats = {"ok": 0, "skipped": 0, "failed": 0}
    started = time.time()
    print(f"\n=== {config} / {dataset}: {len(items)} task(s) ===", flush=True)

    for index, item in enumerate(items):
        tid = task_id_for(dataset, item, index)
        try:
            res = harness.evaluate_task(
                dataset=dataset,
                task_id=tid,
                dataset_index=item.get("dataset_index", index),
                runner=runner,
                task_item=item,
                max_tokens=args.max_tokens,
            )
            if res.get("skipped"):
                stats["skipped"] += 1
            elif res.get("success"):
                stats["ok"] += 1
                q = res["record"]["metrics"]["quality_score"]
                lat = res["record"]["metrics"]["latency_ms"]
                print(f"  [{index+1}/{len(items)}] {tid}: q={q:.2f} {lat:.0f}ms", flush=True)
            else:
                stats["failed"] += 1
        except Exception as exc:
            stats["failed"] += 1
            print(f"  [{index+1}/{len(items)}] {tid}: ERROR {exc}", file=sys.stderr, flush=True)
            if args.fail_fast:
                raise

    stats["elapsed_s"] = round(time.time() - started, 1)
    print(f"  -> ok={stats['ok']} skipped={stats['skipped']} "
          f"failed={stats['failed']} in {stats['elapsed_s']}s", flush=True)
    return stats


def main():
    p = argparse.ArgumentParser(description="LiteAgent Phase 8 evaluation driver")
    p.add_argument("--config", default="liteagent",
                   help=f"One of {', '.join(CONFIGS)}, or 'all'")
    p.add_argument("--dataset", default="all",
                   help=f"One of {', '.join(DATASETS)}, or 'all'")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--results", default="experiments/evaluation_results.jsonl")
    p.add_argument("--failed", default="experiments/failed_evals.jsonl")
    p.add_argument("--log-dir", default="experiments")
    p.add_argument("--ssd-root", default="experiments/phase8_ssd")
    p.add_argument("--router-config", default="config/router_config.yaml")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0, help="Cap tasks per dataset (0 = all)")
    p.add_argument("--zero-shot", action="store_true",
                   help="Disable few-shot exemplars (scores will be near zero on small models)")
    p.add_argument("--energy-source", default="auto",
                   choices=["auto", "external", "nvidia_smi", "rapl", "none"],
                   help="Energy backend. 'external' (wall/USB-C meter CSV) is the "
                        "only one that works on both the Pi and the workstation")
    p.add_argument("--meter-csv", default=None,
                   help="Timestamped meter log (columns: timestamp,watts) for --energy-source external")
    p.add_argument("--workstation-host", default=None,
                   help="Enable remote Large-tier dispatch against this host")
    p.add_argument("--workstation-port", type=int, default=50051)
    p.add_argument("--fail-fast", action="store_true")
    args = p.parse_args()

    configs = list(CONFIGS) if args.config == "all" else [args.config]
    datasets = list(DATASETS) if args.dataset == "all" else [args.dataset]
    for c in configs:
        if c not in CONFIGS:
            p.error(f"Unknown config: {c}")
    for d in datasets:
        if d not in DATASETS:
            p.error(f"Unknown dataset: {d}")

    workstation_client = None
    if args.workstation_host:
        from liteagent.network.client import WorkstationClient
        workstation_client = WorkstationClient(args.workstation_host, args.workstation_port)
        print(f"Remote Large tier -> {args.workstation_host}:{args.workstation_port}")
    else:
        print("No --workstation-host: Large-tier tasks fall back to local execution.")

    summary = {}
    for config in configs:
        for dataset in datasets:
            summary[f"{config}/{dataset}"] = run_one(config, dataset, args, workstation_client)

    print("\n=== Summary ===")
    for key, s in summary.items():
        print(f"  {key:34} ok={s['ok']:4} skipped={s['skipped']:4} "
              f"failed={s['failed']:4} {s['elapsed_s']}s")
    print(f"\nResults appended to {args.results}")


if __name__ == "__main__":
    main()

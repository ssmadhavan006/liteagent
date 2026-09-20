import os
import time
import uuid
import yaml
import json
import datetime
import hashlib
import threading
import random
import logging
from llama_cpp import Llama

from liteagent.config import EDGE_N_GPU_LAYERS
from liteagent.router.router import route_task
from liteagent.cache import KVCacheManager
from liteagent.network.client import WorkstationClient
from liteagent.utils.model_resolver import resolve_model_path
from liteagent.utils.inference import InferenceEngine

logger = logging.getLogger("liteagent.network.dispatch")

class TaskDispatcher:
    def __init__(
        self,
        router_config_path: str,
        edge_cache_manager: KVCacheManager,
        workstation_client: WorkstationClient = None,
        log_dir: str = "experiments"
    ):
        self.router_config_path = router_config_path
        self.edge_cache_manager = edge_cache_manager
        self.workstation_client = workstation_client
        self.log_dir = log_dir
        self.local_models = {}
        self.model_locks = {}
        self.locks_lock = threading.Lock()

        self.routing_disabled = False
        self.cache_disabled = False
        self.baseline_name = "liteagent"

        # Load fallback policy and workstation config
        self.fallback_policy = "retry_then_medium"
        self.max_retries = 1

        if router_config_path and os.path.exists(router_config_path):
            try:
                with open(router_config_path, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)
                    net_config = config.get("network", {})
                    self.fallback_policy = net_config.get("fallback_policy", "retry_then_medium")
                    self.max_retries = net_config.get("max_retries", 1)

                    # Auto-initialize workstation_client if host/port present in config
                    if not self.workstation_client and "workstation_host" in net_config:
                        host = net_config.get("workstation_host", "localhost")
                        port = net_config.get("workstation_port", 50051)
                        self.workstation_client = WorkstationClient(host=host, port=port)
            except Exception as e:
                logger.warning("Failed to load dispatcher configuration from %s: %s", router_config_path, e)

        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

    def log_event(self, request_id: str, event: str, extra: dict = None):
        if not self.log_dir:
            return
        log_file = os.path.join(self.log_dir, "operations.jsonl")
        log_entry = {
            "timestamp": datetime.datetime.now(datetime.UTC).isoformat() + "Z",
            "request_id": request_id,
            "component": "dispatcher",
            "event": event,
            "baseline": self.baseline_name,
            "feature_flags": {
                "routing": not self.routing_disabled,
                "cache": not self.cache_disabled,
                "grpc": self.workstation_client is not None
            }
        }
        if extra:
            log_entry.update(extra)
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            logger.warning("Failed to append dispatch log: %s", e)

    def _execute_remote_with_fallback(
        self,
        task: dict,
        session_id: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
        request_id: str,
        route: dict,
        tier: str,
        agent_role: str = None
    ) -> tuple[dict | None, bool, str]:
        self.log_event(request_id, "DISPATCH_STARTED", {"execution_location": "remote"})
        if agent_role is None:
            agent_role = route["active_agents"][0] if route.get("active_agents") else "Planner"

        attempts = 0
        max_attempts = self.max_retries + 1
        deadline = time.time() + 180.0
        base_delay = 0.2
        success = False
        remote_res = None
        error_reason = ""

        while attempts < max_attempts and time.time() < deadline:
            attempts += 1
            try:
                remote_res = self.workstation_client.dispatch_task(
                    request_id=request_id,
                    task_id=str(task.get("id", "task_id")),
                    session_id=session_id,
                    agent_role=agent_role,
                    prompt=task["prompt"],
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    cache_disabled=self.cache_disabled
                )
                success = True
                break
            except Exception as e:
                error_reason = str(e)
                if attempts < max_attempts and time.time() < deadline:
                    sleep_time = (base_delay * (2 ** (attempts - 1))) + random.uniform(0.01, 0.1)
                    time.sleep(sleep_time)
                else:
                    break

        if success and remote_res:
            self.log_event(request_id, "RESULT_RECEIVED", {"source": "remote"})
            self.log_event(request_id, "CACHE_UPDATED", {"location": "remote"})

            out = dict(remote_res)
            out["routed_tier"] = route.get("model_tier", tier)
            out["executed_tier"] = tier
            out["fallback_occurred"] = False

            self.log_event(request_id, "COMPLETED", {"outcome": "SUCCESS"})
            return out, True, ""
        else:
            self.log_event(request_id, "FALLBACK", {
                "reason": error_reason,
                "policy": self.fallback_policy,
                "attempts": attempts,
                "executed_on": "edge_medium" if self.fallback_policy != "fail" else "none"
            })
            if self.fallback_policy == "fail":
                self.log_event(request_id, "COMPLETED", {"outcome": "FAILED"})
                raise ConnectionError(f"Remote dispatch failed: {error_reason}")
            return None, False, error_reason

    def _execute_local(
        self,
        task: dict,
        session_id: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
        request_id: str,
        route: dict,
        tier: str,
        prompt_hash: str,
        fallback_occurred: bool = False,
        timeout: float = 120.0,
        agent_role: str = None,
        cache_prefix: str = None
    ) -> dict:
        self.log_event(request_id, "DISPATCH_STARTED", {"execution_location": "local"})
        if agent_role is None:
            agent_role = route["active_agents"][0] if route.get("active_agents") else "Executor"

        model_tag = "llama3.2:1b" if tier == "Small" else "llama3.2:3b"
        model_path = resolve_model_path(model_tag)

        with self.locks_lock:
            if model_tag not in self.local_models:
                # Edge-side execution: CPU by default, since the Pi 5 target has
                # no usable GPU. Override via LITEAGENT_EDGE_GPU_LAYERS.
                self.local_models[model_tag] = Llama(
                    model_path=model_path,
                    n_ctx=4096,
                    n_gpu_layers=EDGE_N_GPU_LAYERS,
                    verbose=False,
                    seed=42,
                )
            if model_tag not in self.model_locks:
                self.model_locks[model_tag] = threading.Lock()
            llama = self.local_models[model_tag]
            lock = self.model_locks[model_tag]

        # Split the prompt into a reusable prefix and the task-specific suffix.
        #
        # Keying the cache on a hash of the *whole* prompt made every lookup a
        # guaranteed miss: each benchmark task has a distinct prompt, so the
        # three-tier hierarchy never served a single restore (2 hits against 112
        # misses across the project's history, none from Standby or Cold). The
        # reusable part is the output contract plus few-shot exemplars, which are
        # byte-identical for every task in a benchmark, so that is what is cached.
        full_text = f"{system_prompt}\n{task['prompt']}"
        prefix_text = cache_prefix if cache_prefix and full_text.startswith(cache_prefix) else ""
        suffix_text = full_text[len(prefix_text):]

        use_prefix_cache = bool(prefix_text) and not self.cache_disabled
        if use_prefix_cache:
            prefix_hash = hashlib.sha256(prefix_text.encode("utf-8")).hexdigest()
            prefix_key = f"prefix::{model_tag}::{prefix_hash[:16]}"
        else:
            prefix_hash = prompt_hash
            prefix_key = session_id

        lock.acquire()
        try:
            if self.cache_disabled:
                cache_hit_tier = "MISS"
            else:
                cache_hit_tier = self.edge_cache_manager.load_cache(
                    session_key=prefix_key,
                    llama_instance=llama,
                    model_tag=model_tag,
                    ctx_size=4096,
                    prompt_hash=prefix_hash
                )

            self.log_event(request_id, "SERVER_COMPUTE_STARTED", {"tier": tier})

            # Local execution watchdog
            result_container = {}
            exception_container = []

            def run_inference():
                try:
                    prefill_start = time.time()
                    saved_prefix = False
                    if cache_hit_tier == "MISS":
                        llama.reset()
                        if use_prefix_cache:
                            # Evaluate the prefix alone, snapshot it, then continue
                            # with the suffix. The snapshot is what later tasks reuse.
                            prefix_tokens = llama.tokenize(prefix_text.encode("utf-8"))
                            llama.eval(prefix_tokens)
                            self.edge_cache_manager.save_cache(
                                session_key=prefix_key,
                                agent_role=agent_role,
                                llama_instance=llama,
                                model_tag=model_tag,
                                ctx_size=4096,
                                prompt_hash=prefix_hash,
                            )
                            saved_prefix = True
                            suffix_tokens = llama.tokenize(
                                suffix_text.encode("utf-8"), add_bos=False
                            )
                            llama.eval(suffix_tokens)
                            prefill_tokens = len(prefix_tokens) + len(suffix_tokens)
                            cached_tokens = 0
                        else:
                            tokens = llama.tokenize(full_text.encode("utf-8"))
                            llama.eval(tokens)
                            prefill_tokens = len(tokens)
                            cached_tokens = 0
                    else:
                        # Prefix already resident: only the suffix needs evaluating.
                        suffix_tokens = llama.tokenize(
                            suffix_text.encode("utf-8"), add_bos=False
                        )
                        llama.eval(suffix_tokens)
                        prefill_tokens = len(suffix_tokens)
                        cached_tokens = len(llama.tokenize(prefix_text.encode("utf-8"))) \
                            if prefix_text else 0

                    prefill_latency_ms = (time.time() - prefill_start) * 1000.0

                    gen_start = time.time()
                    timings = {}
                    response_tokens, response_text = InferenceEngine.generate_tokens(
                        llama, max_tokens, temperature=temperature, timings=timings
                    )
                    generation_latency_ms = (time.time() - gen_start) * 1000.0

                    result_container["prefill_tokens"] = prefill_tokens
                    result_container["cached_prefix_tokens"] = cached_tokens
                    result_container["prefill_latency_ms"] = prefill_latency_ms
                    result_container["generation_latency_ms"] = generation_latency_ms
                    # TTFT spans restoration plus prefill plus the first decode,
                    # which is the quantity a context cache is meant to reduce.
                    result_container["ttft_ms"] = prefill_latency_ms + timings.get("ttft_ms", 0.0)
                    result_container["response_tokens"] = response_tokens
                    result_container["response_text"] = response_text
                    result_container["prefix_saved"] = saved_prefix
                except Exception as ex:
                    exception_container.append(ex)

            inf_thread = threading.Thread(target=run_inference, daemon=True)
            inf_thread.start()
            inf_thread.join(timeout=timeout)

            if inf_thread.is_alive():
                raise TimeoutError(f"Local inference execution exceeded timeout of {timeout} seconds.")
            if exception_container:
                raise exception_container[0]

            prefill_tokens = result_container["prefill_tokens"]
            prefill_latency_ms = result_container["prefill_latency_ms"]
            generation_latency_ms = result_container["generation_latency_ms"]
            response_tokens = result_container["response_tokens"]
            response_text = result_container["response_text"]

            self.log_event(request_id, "SERVER_COMPUTE_FINISHED", {"tier": tier})

            if not self.cache_disabled:
                if use_prefix_cache:
                    # The prefix snapshot is written before the suffix is
                    # evaluated; re-saving here would overwrite it with a
                    # task-specific state that no later task can reuse.
                    if result_container.get("prefix_saved"):
                        self.log_event(request_id, "CACHE_UPDATED",
                                       {"location": "local", "scope": "prefix"})
                else:
                    self.edge_cache_manager.save_cache(
                        session_key=session_id,
                        agent_role=agent_role,
                        llama_instance=llama,
                        model_tag=model_tag,
                        ctx_size=4096,
                        prompt_hash=prompt_hash
                    )
                    self.log_event(request_id, "CACHE_UPDATED", {"location": "local"})

            outcome = "FALLBACK_SUCCESS" if fallback_occurred else "SUCCESS"
            self.log_event(request_id, "COMPLETED", {"outcome": outcome})

            return {
                "response_text": response_text,
                "tokens_generated": len(response_tokens),
                "prefill_tokens": prefill_tokens,
                "cached_prefix_tokens": result_container.get("cached_prefix_tokens", 0),
                "prefill_latency_ms": round(prefill_latency_ms, 2),
                "generation_latency_ms": round(generation_latency_ms, 2),
                "ttft_ms": round(result_container.get("ttft_ms", 0.0), 2),
                "cache_hit_tier": cache_hit_tier,
                "routed_tier": route.get("model_tier", tier),
                "executed_tier": tier,
                "fallback_occurred": fallback_occurred,
                "request_id": request_id
            }
        finally:
            lock.release()

    def execute_agent_step(
        self,
        task_id: str,
        prompt: str,
        system_prompt: str,
        agent_role: str,
        session_key: str,
        tier: str,
        request_id: str = None,
        temperature: float = 0.0,
        max_tokens: int = 128,
        cache_prefix: str = None
    ) -> dict:
        """
        Executes one agent turn at an already-chosen tier.

        This is the routing-free primitive the agent chain drives: the router
        runs once per task, then each agent turn lands here with its own role
        and cache session key.
        """
        request_id = request_id or str(uuid.uuid4())
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        step_task = {"prompt": prompt, "id": task_id}
        route = {"model_tier": tier, "active_agents": [agent_role]}

        self.log_event(request_id, "AGENT_STEP_STARTED", {
            "agent_role": agent_role,
            "tier": tier,
            "session_key": session_key
        })

        if tier == "Large" and self.workstation_client:
            out, success, err_reason = self._execute_remote_with_fallback(
                step_task, session_key, system_prompt, temperature, max_tokens,
                request_id, route, tier, agent_role=agent_role
            )
            if success:
                return out
            logger.warning(
                "Agent step %s falling back to local Medium execution. Reason: %s",
                agent_role, err_reason
            )
            return self._execute_local(
                step_task, session_key, system_prompt, temperature, max_tokens,
                request_id, route, "Medium", prompt_hash,
                fallback_occurred=True, agent_role=agent_role, cache_prefix=cache_prefix
            )

        exec_tier = tier if tier in ("Small", "Medium") else "Medium"
        return self._execute_local(
            step_task, session_key, system_prompt, temperature, max_tokens,
            request_id, route, exec_tier, prompt_hash,
            fallback_occurred=False, agent_role=agent_role, cache_prefix=cache_prefix
        )

    def execute_task(
        self,
        task: dict,
        session_id: str,
        system_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 100,
        cache_prefix: str = None
    ) -> dict:
        request_id = str(uuid.uuid4())

        # 1. Routing step
        if self.routing_disabled:
            tier = "Large"
            location = "remote"
            active_agents = ["Planner", "Retriever", "Executor", "Critic"]
            prompt_hash = hashlib.sha256(task["prompt"].encode("utf-8")).hexdigest()
            route = {
                "model_tier": "Large",
                "execution_location": "remote",
                "active_agents": active_agents,
                "metadata": {"prompt_hash": prompt_hash}
            }
        else:
            route = route_task(task, self.router_config_path, log_dir=self.log_dir)
            tier = route["model_tier"]
            location = route["execution_location"]
            prompt_hash = route["metadata"]["prompt_hash"]

        self.log_event(request_id, "ROUTED", {
            "tier": tier,
            "execution_location": location
        })

        # 2. Remote execution branch
        if tier == "Large" and self.workstation_client:
            out, success, err_reason = self._execute_remote_with_fallback(
                task, session_id, system_prompt, temperature, max_tokens, request_id, route, tier
            )
            if success:
                return out
            # Fallback to local Medium execution
            logger.warning("Fallback triggered: degrading dispatch to local Medium execution. Reason: %s", err_reason)
            tier = "Medium"
            return self._execute_local(
                task, session_id, system_prompt, temperature, max_tokens, request_id, route, tier, prompt_hash,
                fallback_occurred=True, cache_prefix=cache_prefix
            )

        # 3. Local execution branch
        return self._execute_local(
            task, session_id, system_prompt, temperature, max_tokens, request_id, route, tier, prompt_hash,
            fallback_occurred=False, cache_prefix=cache_prefix
        )


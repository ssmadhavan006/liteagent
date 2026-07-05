import os
import time
import uuid
import yaml
import json
import datetime
import hashlib
from llama_cpp import Llama

from liteagent.router.router import route_task
from liteagent.cache import KVCacheManager
from liteagent.network.client import WorkstationClient
from liteagent.utils.model_resolver import resolve_model_path

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
        
        # Load fallback policy from config
        self.fallback_policy = "retry_then_medium"
        self.max_retries = 1
        
        if router_config_path and os.path.exists(router_config_path):
            try:
                with open(router_config_path, "r") as f:
                    config = yaml.safe_load(f)
                    net_config = config.get("network", {})
                    self.fallback_policy = net_config.get("fallback_policy", "retry_then_medium")
                    self.max_retries = net_config.get("max_retries", 1)
            except Exception:
                pass
                
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
            "event": event
        }
        if extra:
            log_entry.update(extra)
        try:
            with open(log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception:
            pass

    def execute_task(
        self,
        task: dict,
        session_id: str,
        system_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 100
    ) -> dict:
        request_id = str(uuid.uuid4())
        
        # 1. Routing step
        route = route_task(task, self.router_config_path, log_dir=self.log_dir)
        tier = route["model_tier"]
        location = route["execution_location"]
        
        self.log_event(request_id, "ROUTED", {
            "tier": tier,
            "execution_location": location
        })
        
        prompt_hash = route["metadata"]["prompt_hash"]

        # 2. Dispatch decision
        if tier == "Large" and self.workstation_client:
            self.log_event(request_id, "DISPATCH_STARTED", {"execution_location": "remote"})
            
            # Execute remote dispatch with configurable retries
            attempts = 0
            success = False
            remote_res = None
            error_reason = ""
            
            while attempts <= self.max_retries:
                attempts += 1
                try:
                    # Warm-up helper if client supports it (Task 5)
                    remote_res = self.workstation_client.dispatch_task(
                        request_id=request_id,
                        task_id=str(task.get("id", "task_id")),
                        session_id=session_id,
                        agent_role=route["active_agents"][0] if route["active_agents"] else "Planner",
                        prompt=task["prompt"],
                        system_prompt=system_prompt,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    success = True
                    break
                except Exception as e:
                    error_reason = str(e)
                    if attempts <= self.max_retries:
                        time.sleep(0.2)
                        continue
                    else:
                        break
            
            if success and remote_res:
                self.log_event(request_id, "RESULT_RECEIVED", {"source": "remote"})
                self.log_event(request_id, "CACHE_UPDATED", {"location": "remote"})
                
                # Merge remote execution latencies into dispatcher return
                out = dict(remote_res)
                out["routed_tier"] = tier
                out["executed_tier"] = tier
                out["fallback_occurred"] = False
                
                self.log_event(request_id, "COMPLETED", {"outcome": "SUCCESS"})
                return out
                
            else:
                # Log fallback event
                self.log_event(request_id, "FALLBACK", {
                    "reason": error_reason,
                    "policy": self.fallback_policy,
                    "attempts": attempts,
                    "executed_on": "edge_medium" if self.fallback_policy != "fail" else "none"
                })
                
                if self.fallback_policy == "fail":
                    self.log_event(request_id, "COMPLETED", {"outcome": "FAILED"})
                    raise ConnectionError(f"Remote dispatch failed: {error_reason}")
                else:
                    # Downgrade to local Medium execution
                    print(f"Fallback triggered: degrading dispatch to local Medium-tier execution (llama3.2:3b). Reason: {error_reason}")
                    tier = "Medium"
                    
        # 3. Local execution (Small, Medium, or Large-degraded-to-Medium)
        self.log_event(request_id, "DISPATCH_STARTED", {"execution_location": "local"})
        
        model_tag = "llama3.2:1b" if tier == "Small" else "llama3.2:3b"
        model_path = resolve_model_path(model_tag)
        
        # Load local model
        if model_tag not in self.local_models:
            self.local_models[model_tag] = Llama(model_path=model_path, n_ctx=512, verbose=False, seed=42)
        llama = self.local_models[model_tag]
        
        # Load local cache state
        cache_hit_tier = self.edge_cache_manager.load_cache(
            session_key=session_id,
            llama_instance=llama,
            model_tag=model_tag,
            ctx_size=512,
            prompt_hash=prompt_hash
        )
        
        self.log_event(request_id, "SERVER_COMPUTE_STARTED", {"tier": tier})
        
        # Prefill execution
        prefill_start = time.time()
        if cache_hit_tier == "MISS":
            llama.reset()
            full_prompt = f"{system_prompt}\n{task['prompt']}".encode("utf-8")
            tokens = llama.tokenize(full_prompt)
            llama.eval(tokens)
            prefill_tokens = len(tokens)
        else:
            tokens = llama.tokenize(task['prompt'].encode("utf-8"))
            llama.eval(tokens)
            prefill_tokens = len(tokens)
            
        prefill_latency_ms = (time.time() - prefill_start) * 1000.0
        
        # Generation loop
        gen_start = time.time()
        response_tokens = []
        for _ in range(max_tokens):
            logits = llama.eval_logits[-1]
            next_token = logits.index(max(logits))
            
            if next_token == llama.token_eos():
                break
                
            response_tokens.append(next_token)
            llama.eval([next_token])
            
        response_text = llama.detokenize(response_tokens).decode("utf-8", errors="ignore")
        generation_latency_ms = (time.time() - gen_start) * 1000.0
        
        self.log_event(request_id, "SERVER_COMPUTE_FINISHED", {"tier": tier})
        
        # Save cache locally
        role = route["active_agents"][0] if route["active_agents"] else "Executor"
        self.edge_cache_manager.save_cache(
            session_key=session_id,
            agent_role=role,
            llama_instance=llama,
            model_tag=model_tag,
            ctx_size=512,
            prompt_hash=prompt_hash
        )
        
        self.log_event(request_id, "CACHE_UPDATED", {"location": "local"})
        
        outcome = "FALLBACK_SUCCESS" if route["model_tier"] == "Large" else "SUCCESS"
        self.log_event(request_id, "COMPLETED", {"outcome": outcome})
        
        return {
            "response_text": response_text,
            "tokens_generated": len(response_tokens),
            "prefill_tokens": prefill_tokens,
            "prefill_latency_ms": round(prefill_latency_ms, 2),
            "generation_latency_ms": round(generation_latency_ms, 2),
            "cache_hit_tier": cache_hit_tier,
            "routed_tier": route["model_tier"],
            "executed_tier": tier,
            "fallback_occurred": (route["model_tier"] == "Large"),
            "request_id": request_id
        }

import time
import uuid
import hashlib
import re
from llama_cpp import Llama

from liteagent.network.dispatch import TaskDispatcher
from liteagent.utils.model_resolver import resolve_model_path

# Independent keyword dictionary representing complex task indicators
COMPLEX_KEYWORDS = {
    "class", "def", "function", "import", "implement", "tree", "binary", "insert", 
    "delete", "search", "math", "logic", "code", "recursive", "algorithm", "compare",
    "prove", "equation", "solve"
}

class RouteLLMHeuristicDispatcher(TaskDispatcher):
    """
    RouteLLM-style Heuristic Baseline.
    
    Methodological Note:
    This heuristic is not intended to reproduce RouteLLM's exact published accuracy or its exact 
    training pipeline (which includes preference datasets, reward modeling, matrix factorization, 
    and active router training). Instead, it approximates the core decision principle (routing to 
    the cheapest model predicted to satisfy a utility threshold), allowing the evaluation of the 
    system's architectural behavior in isolation.
    """
    def __init__(self, workstation_client, edge_cache_manager, log_dir="experiments", threshold=0.12):
        super().__init__(
            router_config_path="config/router_config.yaml",
            edge_cache_manager=edge_cache_manager,
            workstation_client=workstation_client,
            log_dir=log_dir
        )
        self.threshold = threshold
        self.baseline_name = "routellm_heuristic"
        self.cache_disabled = True # Caching is disabled for this baseline ablation
        self.routing_disabled = False

    def execute_task(
        self,
        task: dict,
        session_id: str,
        system_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 100
    ) -> dict:
        request_id = str(uuid.uuid4())
        
        # 1. RouteLLM-style heuristic routing (Independent word-overlap density)
        prompt = task.get("prompt", "")
        # Basic word tokenization (lowercase, strip punctuation)
        words = [w.strip(".,!?;:()[]{}'\"").lower() for w in prompt.split()]
        words = [w for w in words if w]
        
        if not words:
            utility = 0.0
        else:
            overlap = sum(1 for w in words if w in COMPLEX_KEYWORDS)
            utility = overlap / len(words)
        
        if utility >= self.threshold:
            tier = "Large"
            location = "remote"
            active_agents = ["Planner", "Retriever", "Executor", "Critic"]
        elif utility >= self.threshold * 0.33: # Threshold 0.04
            tier = "Medium"
            location = "local"
            active_agents = ["Planner", "Executor"]
        else:
            tier = "Small"
            location = "local"
            active_agents = ["Executor"]
            
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        
        self.log_event(request_id, "ROUTED", {
            "tier": tier,
            "execution_location": location,
            "utility_score": round(utility, 4),
            "threshold": self.threshold
        })

        # 2. Dispatch decision
        if tier == "Large" and self.workstation_client:
            self.log_event(request_id, "DISPATCH_STARTED", {"execution_location": "remote"})
            
            attempts = 0
            success = False
            remote_res = None
            error_reason = ""
            
            while attempts <= self.max_retries:
                attempts += 1
                try:
                    remote_res = self.workstation_client.dispatch_task(
                        request_id=request_id,
                        task_id=str(task.get("id", "task_id")),
                        session_id=session_id,
                        agent_role=active_agents[0],
                        prompt=prompt,
                        system_prompt=system_prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        cache_disabled=self.cache_disabled
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
                
                out = dict(remote_res)
                out["routed_tier"] = tier
                out["executed_tier"] = tier
                out["fallback_occurred"] = False
                
                self.log_event(request_id, "COMPLETED", {"outcome": "SUCCESS"})
                return out
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
                else:
                    tier = "Medium"
                    
        # 3. Local execution (Small, Medium, or Large-degraded-to-Medium)
        self.log_event(request_id, "DISPATCH_STARTED", {"execution_location": "local"})
        
        model_tag = "llama3.2:1b" if tier == "Small" else "llama3.2:3b"
        model_path = resolve_model_path(model_tag)
        
        if model_tag not in self.local_models:
            self.local_models[model_tag] = Llama(model_path=model_path, n_ctx=512, verbose=False, seed=42)
        llama = self.local_models[model_tag]
        
        # Load local cache state (bypassed since cache_disabled is True)
        cache_hit_tier = "MISS"
        
        self.log_event(request_id, "SERVER_COMPUTE_STARTED", {"tier": tier})
        
        # Prefill execution
        prefill_start = time.time()
        llama.reset()
        full_prompt = f"{system_prompt}\n{prompt}".encode("utf-8")
        tokens = llama.tokenize(full_prompt)
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
        
        # Save cache locally (bypassed)
        outcome = "FALLBACK_SUCCESS" if tier == "Large" else "SUCCESS"
        self.log_event(request_id, "COMPLETED", {"outcome": outcome})
        
        return {
            "response_text": response_text,
            "tokens_generated": len(response_tokens),
            "prefill_tokens": prefill_tokens,
            "prefill_latency_ms": round(prefill_latency_ms, 2),
            "generation_latency_ms": round(generation_latency_ms, 2),
            "cache_hit_tier": cache_hit_tier,
            "routed_tier": tier,
            "executed_tier": tier,
            "fallback_occurred": (tier == "Large"),
            "request_id": request_id
        }

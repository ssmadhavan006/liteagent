import uuid
import hashlib

from liteagent.network.dispatch import TaskDispatcher

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

        # 2. Dispatch decision using inherited execution helpers
        route = {
            "model_tier": tier,
            "execution_location": location,
            "active_agents": active_agents,
            "metadata": {"prompt_hash": prompt_hash}
        }

        if tier == "Large" and self.workstation_client:
            out, success, err_reason = self._execute_remote_with_fallback(
                task, session_id, system_prompt, temperature, max_tokens, request_id, route, tier
            )
            if success:
                return out
            # Fallback to local Medium execution
            tier = "Medium"
            return self._execute_local(
                task, session_id, system_prompt, temperature, max_tokens, request_id, route, tier, prompt_hash, fallback_occurred=True
            )

        # 3. Local execution branch
        return self._execute_local(
            task, session_id, system_prompt, temperature, max_tokens, request_id, route, tier, prompt_hash, fallback_occurred=False
        )

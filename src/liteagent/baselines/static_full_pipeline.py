import os
from liteagent.network.dispatch import TaskDispatcher
from liteagent.cache import KVCacheManager

class StaticFullPipelineRunner:
    """
    Static Full-Pipeline Baseline.
    Statically maps all tasks to the Large tier ( llama3.1:8b ) on the Workstation via gRPC.
    Caching is completely disabled, forcing a full prompt prefill on every step.
    """
    def __init__(self, workstation_client, log_dir="experiments"):
        # We pass a dummy KVCacheManager since caching is fully bypassed
        dummy_cache_manager = KVCacheManager(
            max_ram_states=1,
            ssd_dir=os.path.join(log_dir, "dummy_ssd") if log_dir else "dummy_ssd",
            log_dir=log_dir
        )
        self.dispatcher = TaskDispatcher(
            router_config_path="config/router_config.yaml",
            edge_cache_manager=dummy_cache_manager,
            workstation_client=workstation_client,
            log_dir=log_dir
        )
        self.dispatcher.routing_disabled = True
        self.dispatcher.cache_disabled = True
        self.dispatcher.baseline_name = "static_full"

    def execute_task(self, task, session_id, system_prompt="", temperature=0.0, max_tokens=100):
        return self.dispatcher.execute_task(
            task=task,
            session_id=session_id,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )

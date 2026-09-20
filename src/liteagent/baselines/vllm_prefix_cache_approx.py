import os
from liteagent.cache.manager import KVCacheManager

class VLLMPrefixCacheApprox(KVCacheManager):
    """
    vLLM-inspired single-tier in-memory prefix cache approximation.
    Keeps states strictly in active RAM/VRAM with no swapping/eviction to secondary storage.
    Enforces a strict slot-capacity ceiling to test exhaustion limit behaviors.
    """
    def __init__(self, max_in_memory_slots=2, log_dir="experiments"):
        # We pass a temporary SSD path to the parent but override load/save to bypass disk usage entirely
        super().__init__(
            max_ram_states=max_in_memory_slots,
            ssd_dir=os.path.join(log_dir, "temp_vllm_ssd") if log_dir else "temp_vllm_ssd",
            log_dir=log_dir
        )
        self.max_slots = max_in_memory_slots
        self.active_slots = {} # session_key -> state object

    def load_cache(self, session_key: str, llama_instance, model_tag: str, ctx_size: int, prompt_hash: str) -> str:
        # Check active in-memory slots only
        if session_key in self.active_slots:
            state = self.active_slots[session_key]
            llama_instance.load_state(state)
            self._write_log({
                "component": "cache_manager",
                "event": "CACHE_HIT",
                "session_key": session_key,
                "tier": "HOT"
            })
            return "HOT"
        else:
            self._write_log({
                "component": "cache_manager",
                "event": "CACHE_MISS",
                "session_key": session_key
            })
            return "MISS"

    def save_cache(self, session_key: str, agent_role: str, llama_instance, model_tag: str, ctx_size: int, prompt_hash: str):
        # Enforce slot capacity bounds
        if session_key not in self.active_slots:
            if len(self.active_slots) >= self.max_slots:
                self._write_log({
                    "component": "cache_manager",
                    "event": "EXPECTED_LIMIT_REACHED",
                    "reason": "Max in-memory prefix slots exhausted.",
                    "session_key": session_key
                })
                raise RuntimeError("EXPECTED_LIMIT_REACHED: Max in-memory prefix slots exhausted.")

        try:
            state = llama_instance.save_state()
            self.active_slots[session_key] = state
            self._write_log({
                "component": "cache_manager",
                "event": "CACHE_SAVE",
                "session_key": session_key,
                "tier": "HOT"
            })
        except Exception as e:
            self._write_log({
                "component": "cache_manager",
                "event": "FAILURE_ALLOCATION",
                "error": str(e),
                "session_key": session_key
            })
            raise RuntimeError(f"FAILURE_ALLOCATION: {e}") from e

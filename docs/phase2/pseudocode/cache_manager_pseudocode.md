# LiteAgent — Cache Manager Pseudocode

This document contains the algorithmic pseudocode for the Three-Tier KV-Cache Manager.

```python
# Three-Tier KV-Cache Manager Algorithm

class CacheStateMetadata:
    def __init__(self, session_key: str, agent_role: str):
        self.session_key = session_key
        self.agent_role = agent_role
        self.last_accessed = time.time()

class ThreeTierCacheManager:
    def __init__(self, max_ram_states: int, ssd_dir: str):
        self.max_ram_states = max_ram_states
        self.ssd_dir = ssd_dir
        
        # In-memory storage tiers
        self.hot_state = None            # Active slot memory of Llama instance
        self.standby_cache = {}          # Dict[session_key, bytes] (Host RAM)
        self.metadata_store = {}         # Dict[session_key, CacheStateMetadata]
        
        # Static role priorities (PW-LRU)
        self.role_priorities = {
            "Critic": 1.0,
            "Executor": 0.8,
            "Planner": 0.5,
            "Retriever": 0.3
        }

    def compute_cache_key(self, model_name: str, system_prompt: str, context_history: list[str]) -> str:
        history_str = "".join(context_history)
        raw_key = f"{model_name}||{system_prompt}||{history_str}"
        return sha256_hash(raw_key)

    def load_cache(self, session_key: str, llama_instance) -> str:
        """
        Restores KV cache and returns hit status: HOT, STANDBY, COLD, or MISS
        """
        # Tier 1 check (Hot)
        if self.hot_state == session_key:
            self.metadata_store[session_key].last_accessed = time.time()
            return "HOT"
            
        # Tier 2 check (Standby RAM)
        if session_key in self.standby_cache:
            state_bytes = self.standby_cache[session_key]
            llama_instance.load_state(state_bytes)
            self.hot_state = session_key
            self.metadata_store[session_key].last_accessed = time.time()
            return "STANDBY"
            
        # Tier 3 check (Cold SSD)
        ssd_path = f"{self.ssd_dir}/{session_key}.bin"
        if file_exists(ssd_path):
            state_bytes = read_file_bytes(ssd_path)
            llama_instance.load_state(state_bytes)
            self.hot_state = session_key
            
            # Promote to Tier 2 (Standby)
            self.promote_to_standby(session_key, state_bytes)
            self.metadata_store[session_key].last_accessed = time.time()
            return "COLD"
            
        return "MISS"

    def promote_to_standby(self, session_key: str, state_bytes: bytes):
        if len(self.standby_cache) >= self.max_ram_states:
            # Find candidate for eviction
            self.evict_standby_to_cold()
            
        self.standby_cache[session_key] = state_bytes

    def evict_standby_to_cold(self):
        """
        Evicts a standby RAM state to SSD disk based on PW-LRU score
        """
        current_time = time.time()
        
        highest_score = -1.0
        victim_key = None
        
        for key, meta in self.metadata_store.items():
            if key not in self.standby_cache:
                continue
                
            t_elapsed = current_time - meta.last_accessed
            
            # Lookup role priority weight
            w_agent = self.role_priorities.get(meta.agent_role, 0.0)
            
            # Compute virtual PW-LRU score
            score = t_elapsed * (1.0 - w_agent)
            
            if score > highest_score:
                highest_score = score
                victim_key = key
                
        if victim_key:
            # Write bytes to NVMe SSD
            victim_bytes = self.standby_cache[victim_key]
            write_file_bytes(f"{self.ssd_dir}/{victim_key}.bin", victim_bytes)
            
            # Remove from Host RAM
            del self.standby_cache[victim_key]

    def save_cache(self, session_key: str, agent_role: str, llama_instance):
        """
        Extracts context state from model and updates cache stores
        """
        state_bytes = llama_instance.save_state()
        
        # Save to RAM
        self.promote_to_standby(session_key, state_bytes)
        
        # Record metadata
        self.metadata_store[session_key] = CacheStateMetadata(session_key, agent_role)
        self.hot_state = session_key
```

import os
import time
import json
import hashlib
import datetime
import threading
from liteagent.cache.tiers import CacheStateMetadata
from liteagent.cache.eviction import select_eviction_victim
from liteagent.cache.serialization import serialize_llama_state, deserialize_llama_state, SerializationError
from liteagent.cache.storage import StorageManager, CacheRestoreError

class KVCacheManager:
    def __init__(self, max_ram_states: int, ssd_dir: str, log_dir: str = "experiments"):
        self.max_ram_states = max_ram_states
        self.ssd_dir = ssd_dir
        self.log_dir = log_dir

        self.storage = StorageManager(ssd_dir)
        self.hot_state = None
        # The HOT path returns without deserialising, on the assumption that the
        # model still holds that exact context. Generation invalidates that
        # assumption: the context has advanced by the generated tokens. Serving
        # HOT afterwards silently continues from a polluted context and produces
        # different output, which the Exp 2 losslessness check detects. Callers
        # mark the context dirty once they mutate it.
        self._hot_dirty = False
        self.metadata_store = {}  # Dict[key, CacheStateMetadata]
        self._lock = threading.RLock()

        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

    def compute_cache_key(self, model_name: str, system_prompt: str, context_history: list[str]) -> str:
        history_str = "".join(context_history)
        raw_key = f"{model_name}||{system_prompt}||{history_str}"
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    def _write_log(self, log_entry: dict):
        if not self.log_dir:
            return
        log_file = os.path.join(self.log_dir, "cache_operations.jsonl")
        log_entry["timestamp"] = datetime.datetime.now(datetime.UTC).isoformat() + "Z"
        try:
            with open(log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception:
            pass

    def load_cache(self, session_key: str, llama_instance, model_tag: str, ctx_size: int, prompt_hash: str) -> str:
        """
        Restores KV cache and returns hit status: HOT, STANDBY, COLD, or MISS.
        Raises CacheRestoreError on explicit failure.
        """
        with self._lock:
            # Validate prompt_hash and model_tag if metadata exists
            if session_key in self.metadata_store:
                stored_meta = self.metadata_store[session_key]
                if stored_meta.prompt_hash and stored_meta.prompt_hash != prompt_hash:
                    self._write_log({
                        "event": "PROMPT_MISMATCH_MISS",
                        "session_key": session_key,
                        "stored_hash": stored_meta.prompt_hash,
                        "request_hash": prompt_hash
                    })
                    return "MISS"
                if stored_meta.model_tag and stored_meta.model_tag != model_tag:
                    self._write_log({
                        "event": "MODEL_MISMATCH_MISS",
                        "session_key": session_key,
                        "stored_model": stored_meta.model_tag,
                        "request_model": model_tag
                    })
                    return "MISS"

            # 1. Tier 1 (Hot) Check. Only valid while the in-memory context is
            # still byte-identical to what was saved; otherwise fall through to
            # Standby, which performs a real restore.
            if self.hot_state == session_key and not self._hot_dirty:
                if session_key in self.metadata_store:
                    self.metadata_store[session_key].last_accessed = time.time()
                self._write_log({
                    "event": "LOAD_HIT",
                    "tier": "HOT",
                    "session_key": session_key,
                    "model_tag": model_tag,
                    "prompt_hash": prompt_hash
                })
                return "HOT"

            # 2. Tier 2 (Standby RAM) Check
            if session_key in self.storage.standby_cache:
                try:
                    state_bytes = self.storage.load_from_standby(session_key)
                    load_ms = deserialize_llama_state(llama_instance, state_bytes)
                    self.hot_state = session_key
                    self._hot_dirty = False
                    if session_key in self.metadata_store:
                        self.metadata_store[session_key].last_accessed = time.time()

                    self._write_log({
                        "event": "LOAD_HIT",
                        "tier": "STANDBY",
                        "session_key": session_key,
                        "model_tag": model_tag,
                        "prompt_hash": prompt_hash,
                        "load_state_ms": round(load_ms, 2)
                    })
                    return "STANDBY"
                except (SerializationError, CacheRestoreError) as e:
                    # Purge corrupt standby entry and metadata
                    self.storage.delete_from_standby(session_key)
                    self.metadata_store.pop(session_key, None)
                    self._write_log({
                        "event": "RESTORE_FAILED",
                        "tier": "STANDBY",
                        "session_key": session_key,
                        "error": str(e)
                    })
                    raise CacheRestoreError(f"CACHE_RESTORE_FAILED: Standby load error: {e}") from e

            # 3. Tier 3 (Cold SSD) Check
            ssd_bin_path = os.path.join(self.ssd_dir, f"{session_key}.bin") if self.ssd_dir else ""
            if ssd_bin_path and os.path.exists(ssd_bin_path):
                try:
                    state_bytes, meta_dict, disk_read_ms = self.storage.load_from_ssd(session_key)

                    # Validate model tag and prompt hash compatibility from cold metadata
                    if meta_dict.get("model_tag") != model_tag:
                        raise CacheRestoreError(f"Model tag mismatch: stored {meta_dict.get('model_tag')} != {model_tag}")
                    if meta_dict.get("prompt_hash") and meta_dict.get("prompt_hash") != prompt_hash:
                        raise CacheRestoreError("Prompt hash mismatch in SSD cache metadata")

                    load_ms = deserialize_llama_state(llama_instance, state_bytes)
                    self.hot_state = session_key
                    self._hot_dirty = False

                    # Restore metadata object in coordinator
                    meta = CacheStateMetadata.from_dict(meta_dict)
                    meta.session_key = session_key
                    self.metadata_store[session_key] = meta

                    # Promote to Tier 2 (Standby)
                    self.promote_to_standby(session_key, state_bytes, meta)
                    self.metadata_store[session_key].last_accessed = time.time()

                    self._write_log({
                        "event": "LOAD_HIT",
                        "tier": "COLD",
                        "session_key": session_key,
                        "model_tag": model_tag,
                        "prompt_hash": prompt_hash,
                        "disk_read_ms": round(disk_read_ms, 2),
                        "load_state_ms": round(load_ms, 2)
                    })
                    return "COLD"
                except Exception as e:
                    self._write_log({
                        "event": "RESTORE_FAILED",
                        "tier": "COLD",
                        "session_key": session_key,
                        "error": str(e)
                    })
                    raise CacheRestoreError(f"CACHE_RESTORE_FAILED: Cold load error: {e}") from e

            # 4. Cache Miss
            self._write_log({
                "event": "LOAD_MISS",
                "session_key": session_key,
                "model_tag": model_tag,
                "prompt_hash": prompt_hash
            })
            return "MISS"

    def save_cache(self, session_key: str, agent_role: str, llama_instance, model_tag: str, ctx_size: int, prompt_hash: str):
        """
        Saves current Llama state to Standby cache and updates metadata.
        """
        with self._lock:
            try:
                state_bytes, save_ms = serialize_llama_state(llama_instance)
            except SerializationError as e:
                self._write_log({
                    "event": "SAVE_FAILED",
                    "session_key": session_key,
                    "error": str(e)
                })
                raise CacheRestoreError(f"CACHE_SAVE_FAILED: {e}") from e

            meta = CacheStateMetadata(
                session_key=session_key,
                agent_role=agent_role,
                model_tag=model_tag,
                ctx_size=ctx_size,
                prompt_hash=prompt_hash,
                state_size_bytes=len(state_bytes)
            )

            # Save to memory cache
            self.promote_to_standby(session_key, state_bytes, meta)
            self.hot_state = session_key
            self._hot_dirty = False

            self._write_log({
                "event": "SAVE",
                "session_key": session_key,
                "agent_role": agent_role,
                "model_tag": model_tag,
                "save_state_ms": round(save_ms, 2),
                "state_size_bytes": len(state_bytes)
            })

    def mark_context_dirty(self):
        """
        Declares that the model's context no longer matches the saved HOT state.

        Call this after generating, or after evaluating tokens on top of a
        restored context. Without it the next HOT hit resumes from the advanced
        context instead of the cached one.
        """
        with self._lock:
            self._hot_dirty = True

    def promote_to_standby(self, session_key: str, state_bytes: bytes, metadata: CacheStateMetadata):
        """
        Promotes context state to Standby cache, running eviction checks if limit is exceeded.
        """
        with self._lock:
            # Only evict if key is new to standby and we are at capacity
            if session_key not in self.storage.standby_cache:
                if len(self.storage.standby_cache) >= self.max_ram_states:
                    self.evict_standby_to_cold()

            self.storage.save_to_standby(session_key, state_bytes)
            self.metadata_store[session_key] = metadata

    def evict_standby_to_cold(self):
        """
        Evicts a standby context to SSD according to PW-LRU.
        """
        with self._lock:
            standby_keys = list(self.storage.standby_cache.keys())
            if not standby_keys:
                return

            victim_key, score = select_eviction_victim(self.metadata_store, standby_keys)
            if not victim_key or victim_key not in self.metadata_store:
                if standby_keys:
                    victim_key = standby_keys[0]
                else:
                    return

            victim_meta = self.metadata_store.get(victim_key)
            victim_bytes = self.storage.load_from_standby(victim_key)

            if not victim_meta:
                # Log drop and cleanup
                self.storage.delete_from_standby(victim_key)
                self.metadata_store.pop(victim_key, None)
                self._write_log({
                    "event": "EVICT_DROPPED_NO_METADATA",
                    "session_key": victim_key
                })
                return

            # Save to cold SSD
            disk_write_ms, size_bytes = self.storage.save_to_ssd(
                victim_key,
                victim_bytes,
                victim_meta.to_dict()
            )

            # Remove from standby RAM
            self.storage.delete_from_standby(victim_key)

            self._write_log({
                "event": "EVICT",
                "session_key": victim_key,
                "agent_role": victim_meta.agent_role,
                "reason": "PW_LRU_SCORE",
                "score": round(score, 2),
                "destination": "Cold SSD",
                "disk_write_ms": round(disk_write_ms, 2),
                "state_size_bytes": size_bytes
            })

import os
import json
import time

class CacheRestoreError(Exception):
    pass

class StorageManager:
    def __init__(self, ssd_dir: str):
        self.ssd_dir = ssd_dir
        self.standby_cache = {}  # Dict[key, bytes]
        
        if ssd_dir:
            os.makedirs(ssd_dir, exist_ok=True)

    def save_to_standby(self, key: str, data: bytes):
        self.standby_cache[key] = data

    def load_from_standby(self, key: str) -> bytes:
        if key not in self.standby_cache:
            raise CacheRestoreError(f"Key {key} not found in Standby RAM cache.")
        return self.standby_cache[key]

    def delete_from_standby(self, key: str):
        if key in self.standby_cache:
            del self.standby_cache[key]

    def save_to_ssd(self, key: str, data: bytes, metadata: dict) -> tuple[float, int]:
        """
        Saves binary state and companion JSON metadata to cold SSD storage.
        Returns (disk_write_ms, state_size_bytes).
        """
        if not self.ssd_dir:
            raise ValueError("SSD directory not configured.")
            
        start_time = time.perf_counter()
        
        bin_path = os.path.join(self.ssd_dir, f"{key}.bin")
        json_path = os.path.join(self.ssd_dir, f"{key}.json")
        
        try:
            # Write state binary
            with open(bin_path, "wb") as f:
                f.write(data)
                
            # Write companion JSON metadata
            with open(json_path, "w") as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            # Cleanup partial writes
            if os.path.exists(bin_path):
                os.remove(bin_path)
            if os.path.exists(json_path):
                os.remove(json_path)
            raise IOError(f"SSD Write failed for key {key}: {e}") from e
            
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return latency_ms, len(data)

    def load_from_ssd(self, key: str) -> tuple[bytes, dict, float]:
        """
        Loads binary state and JSON metadata from cold SSD storage.
        Returns (data_bytes, metadata_dict, disk_read_ms).
        Raises CacheRestoreError on missing or corrupted files.
        """
        if not self.ssd_dir:
            raise ValueError("SSD directory not configured.")
            
        bin_path = os.path.join(self.ssd_dir, f"{key}.bin")
        json_path = os.path.join(self.ssd_dir, f"{key}.json")
        
        # 1. Missing file checks
        if not os.path.exists(bin_path):
            raise CacheRestoreError(f"CACHE_RESTORE_FAILED: Binary cache file missing at {bin_path}")
        if not os.path.exists(json_path):
            raise CacheRestoreError(f"CACHE_RESTORE_FAILED: Metadata cache file missing at {json_path}")
            
        # 2. Corruption checks: zero length
        if os.path.getsize(bin_path) == 0:
            raise CacheRestoreError(f"CACHE_RESTORE_FAILED: Binary cache file at {bin_path} is empty")
        if os.path.getsize(json_path) == 0:
            raise CacheRestoreError(f"CACHE_RESTORE_FAILED: Metadata cache file at {json_path} is empty")
            
        start_time = time.perf_counter()
        
        try:
            with open(bin_path, "rb") as f:
                data = f.read()
        except Exception as e:
            raise CacheRestoreError(f"CACHE_RESTORE_FAILED: Binary read error at {bin_path}: {e}") from e
            
        try:
            with open(json_path, "r") as f:
                metadata = json.load(f)
        except Exception as e:
            raise CacheRestoreError(f"CACHE_RESTORE_FAILED: Metadata corruption (invalid JSON) at {json_path}: {e}") from e
            
        # Check for partial write indicator (e.g. metadata is missing required fields)
        if "model_tag" not in metadata or "state_size_bytes" not in metadata:
            raise CacheRestoreError(f"CACHE_RESTORE_FAILED: Metadata incomplete or corrupted at {json_path}")
            
        # Check size consistency
        if len(data) != metadata["state_size_bytes"]:
            raise CacheRestoreError(
                f"CACHE_RESTORE_FAILED: Size mismatch between binary file ({len(data)}B) and metadata ({metadata['state_size_bytes']}B)"
            )
            
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return data, metadata, latency_ms

    def delete_from_ssd(self, key: str):
        if not self.ssd_dir:
            return
        bin_path = os.path.join(self.ssd_dir, f"{key}.bin")
        json_path = os.path.join(self.ssd_dir, f"{key}.json")
        
        if os.path.exists(bin_path):
            try:
                os.remove(bin_path)
            except OSError:
                pass
        if os.path.exists(json_path):
            try:
                os.remove(json_path)
            except OSError:
                pass

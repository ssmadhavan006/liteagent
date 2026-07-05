import os
import shutil
import pytest
import json
from liteagent.cache.storage import StorageManager, CacheRestoreError

SSD_TEST_DIR = "tests/cache/temp_ssd"

@pytest.fixture(autouse=True)
def setup_and_teardown():
    # Clean up before test
    if os.path.exists(SSD_TEST_DIR):
        shutil.rmtree(SSD_TEST_DIR)
    yield
    # Clean up after test
    if os.path.exists(SSD_TEST_DIR):
        shutil.rmtree(SSD_TEST_DIR)

def test_ram_storage():
    sm = StorageManager(ssd_dir=None)
    sm.save_to_standby("session1", b"dummy_bytes")
    assert sm.load_from_standby("session1") == b"dummy_bytes"
    
    sm.delete_from_standby("session1")
    with pytest.raises(CacheRestoreError):
        sm.load_from_standby("session1")

def test_ssd_storage_success():
    sm = StorageManager(ssd_dir=SSD_TEST_DIR)
    meta = {
        "cache_format_version": 1,
        "llama_cpp_version": "0.3.1",
        "model_tag": "llama3.2:1b",
        "ctx_size": 512,
        "agent_role": "Planner",
        "created_at": "2026-07-05T12:00:00Z",
        "prompt_hash": "hash123",
        "state_size_bytes": 11
    }
    data = b"hello world"
    
    write_ms, size = sm.save_to_ssd("session1", data, meta)
    assert write_ms >= 0.0
    assert size == 11
    
    loaded_data, loaded_meta, read_ms = sm.load_from_ssd("session1")
    assert loaded_data == data
    assert loaded_meta["agent_role"] == "Planner"
    assert read_ms >= 0.0

def test_ssd_failure_missing_files():
    sm = StorageManager(ssd_dir=SSD_TEST_DIR)
    with pytest.raises(CacheRestoreError) as exc_info:
        sm.load_from_ssd("non_existent_key")
    assert "CACHE_RESTORE_FAILED" in str(exc_info.value)

def test_ssd_failure_empty_files():
    sm = StorageManager(ssd_dir=SSD_TEST_DIR)
    bin_path = os.path.join(SSD_TEST_DIR, "session_empty.bin")
    json_path = os.path.join(SSD_TEST_DIR, "session_empty.json")
    
    os.makedirs(SSD_TEST_DIR, exist_ok=True)
    open(bin_path, "wb").close() # Create 0-byte file
    with open(json_path, "w") as f:
        json.dump({"dummy": "value"}, f)
        
    with pytest.raises(CacheRestoreError) as exc_info:
        sm.load_from_ssd("session_empty")
    assert "empty" in str(exc_info.value)

def test_ssd_failure_corrupted_json():
    sm = StorageManager(ssd_dir=SSD_TEST_DIR)
    bin_path = os.path.join(SSD_TEST_DIR, "session_corrupt.bin")
    json_path = os.path.join(SSD_TEST_DIR, "session_corrupt.json")
    
    os.makedirs(SSD_TEST_DIR, exist_ok=True)
    with open(bin_path, "wb") as f:
        f.write(b"some data")
    with open(json_path, "w") as f:
        f.write("invalid json string {:")
        
    with pytest.raises(CacheRestoreError) as exc_info:
        sm.load_from_ssd("session_corrupt")
    assert "invalid JSON" in str(exc_info.value)

def test_ssd_failure_size_mismatch():
    sm = StorageManager(ssd_dir=SSD_TEST_DIR)
    meta = {
        "model_tag": "llama3.2:1b",
        "state_size_bytes": 99999 # size mismatch
    }
    bin_path = os.path.join(SSD_TEST_DIR, "session_mismatch.bin")
    json_path = os.path.join(SSD_TEST_DIR, "session_mismatch.json")
    
    os.makedirs(SSD_TEST_DIR, exist_ok=True)
    with open(bin_path, "wb") as f:
        f.write(b"small data")
    with open(json_path, "w") as f:
        json.dump(meta, f)
        
    with pytest.raises(CacheRestoreError) as exc_info:
        sm.load_from_ssd("session_mismatch")
    assert "mismatch" in str(exc_info.value).lower()

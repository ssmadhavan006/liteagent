import os
import json
import hashlib
from liteagent.router import route_task

def test_route_task_integration():
    config_path = "config/router_config.yaml"
    task = {
        "prompt": "What is 15 + 23?",
        "benchmark": "GSM8K",
        "metadata": {}
    }
    
    # We specify a temporary log folder to isolate test output
    test_log_dir = "tests/router/test_logs"
    os.makedirs(test_log_dir, exist_ok=True)
    log_file = os.path.join(test_log_dir, "routing_decisions.jsonl")
    if os.path.exists(log_file):
        os.remove(log_file)
        
    result = route_task(task, config_path=config_path, log_dir=test_log_dir)
    
    assert "model_tier" in result
    assert "execution_location" in result
    assert "active_agents" in result
    assert "pruned_agents" in result
    assert "metadata" in result
    
    # Verify hash match
    expected_hash = hashlib.sha256(task["prompt"].encode("utf-8")).hexdigest()
    assert result["metadata"]["prompt_hash"] == expected_hash
    
    # Verify log file was written and is valid JSONL
    assert os.path.exists(log_file)
    with open(log_file, "r") as f:
        log_lines = f.readlines()
        
    assert len(log_lines) == 1
    log_data = json.loads(log_lines[0])
    assert log_data["prompt_hash"] == expected_hash
    assert log_data["tier"] == result["model_tier"]
    assert log_data["routing_margin"] == round(result["metadata"]["routing_margin"], 4)
    
    # Cleanup
    if os.path.exists(log_file):
        os.remove(log_file)
    if os.path.exists(test_log_dir):
        os.rmdir(test_log_dir)

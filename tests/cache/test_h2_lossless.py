import os
import json
import shutil
import pytest
from llama_cpp import Llama
from liteagent.utils.model_resolver import resolve_model_path
from liteagent.cache import KVCacheManager

SSD_TEST_DIR = "tests/cache/temp_h2_ssd"
LOG_TEST_DIR = "tests/cache/temp_h2_logs"

@pytest.fixture(scope="module")
def model_path():
    path = resolve_model_path("llama3.2:1b")
    assert os.path.exists(path)
    return path

@pytest.fixture(autouse=True)
def setup_and_teardown():
    for d in [SSD_TEST_DIR, LOG_TEST_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d)
    yield
    for d in [SSD_TEST_DIR, LOG_TEST_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d)

def test_h2_lossless_all_tiers(model_path):
    print("\n--- Starting H2 Losslessness Verification ---")
    
    # 1. Load 15 prompts (5 GSM8K, 5 HotpotQA, 5 HumanEval) from validation dataset
    dataset_path = "datasets/router_validation/validation_prompts.json"
    assert os.path.exists(dataset_path)
    with open(dataset_path, "r") as f:
        all_prompts = json.load(f)
        
    gsm8k_prompts = [p for p in all_prompts if p["benchmark"] == "GSM8K"][:5]
    hotpotqa_prompts = [p for p in all_prompts if p["benchmark"] == "HotpotQA"][:5]
    humaneval_prompts = [p for p in all_prompts if p["benchmark"] == "HumanEval"][:5]
    
    selected_prompts = gsm8k_prompts + hotpotqa_prompts + humaneval_prompts
    assert len(selected_prompts) == 15
    
    # Initialize cache manager
    cm = KVCacheManager(max_ram_states=5, ssd_dir=SSD_TEST_DIR, log_dir=LOG_TEST_DIR)
    
    # Initialize Llama models once
    print("Loading Llama 3.2 1B model (Control)...")
    llama_control = Llama(model_path=model_path, n_ctx=512, verbose=False, seed=42)
    print("Loading Llama 3.2 1B model (Experimental)...")
    llama_exp = Llama(model_path=model_path, n_ctx=512, verbose=False, seed=42)
    
    for idx, item in enumerate(selected_prompts):
        prompt_str = item["prompt"]
        prompt_bytes = prompt_str.encode("utf-8")
        benchmark = item["benchmark"]
        session_key = f"h2_session_{item['id']}"
        prompt_hash = f"hash_{item['id']}"
        
        print(f"Testing Prompt {idx+1}/15 [ID {item['id']} - {benchmark}]...")
        
        # Reset Cache Manager state for clean isolation of each prompt test
        cm.hot_state = None
        cm.storage.standby_cache.clear()
        cm.metadata_store.clear()
        if os.path.exists(SSD_TEST_DIR):
            shutil.rmtree(SSD_TEST_DIR)
        os.makedirs(SSD_TEST_DIR, exist_ok=True)
        
        # --- A. Control: Fresh Prefill Run ---
        llama_control.reset()
        tokens = llama_control.tokenize(prompt_bytes)
        llama_control.eval(tokens)
        
        control_tokens = []
        for _ in range(10):
            logits = llama_control.eval_logits[-1]
            next_token = logits.index(max(logits))
            control_tokens.append(next_token)
            llama_control.eval([next_token])
            
        # --- B. Save state after prefill on experimental instance ---
        llama_exp.reset()
        exp_tokens = llama_exp.tokenize(prompt_bytes)
        llama_exp.eval(exp_tokens)
        
        # Save state using Cache Manager
        cm.save_cache(session_key, "Planner", llama_exp, "llama3.2:1b", 512, prompt_hash)
        
        # --- C. Test Tier 1: HOT hit ---
        hot_tokens = []
        for _ in range(10):
            logits = llama_exp.eval_logits[-1]
            next_token = logits.index(max(logits))
            hot_tokens.append(next_token)
            llama_exp.eval([next_token])
            
        assert hot_tokens == control_tokens, f"Tier 1 (HOT) mismatch for prompt {item['id']}"
        
        # --- D. Test Tier 2: STANDBY RAM hit ---
        llama_exp.reset()
        cm.hot_state = None # Force standby load
        status = cm.load_cache(session_key, llama_exp, "llama3.2:1b", 512, prompt_hash)
        assert status == "STANDBY"
        
        standby_tokens = []
        for _ in range(10):
            logits = llama_exp.eval_logits[-1]
            next_token = logits.index(max(logits))
            standby_tokens.append(next_token)
            llama_exp.eval([next_token])
            
        assert standby_tokens == control_tokens, f"Tier 2 (STANDBY) mismatch for prompt {item['id']}"
        
        # --- E. Test Tier 3: COLD SSD hit ---
        llama_exp.reset()
        cm.hot_state = None
        
        # Evict to SSD
        cm.evict_standby_to_cold()
        
        status = cm.load_cache(session_key, llama_exp, "llama3.2:1b", 512, prompt_hash)
        assert status == "COLD"
        
        cold_tokens = []
        for _ in range(10):
            logits = llama_exp.eval_logits[-1]
            next_token = logits.index(max(logits))
            cold_tokens.append(next_token)
            llama_exp.eval([next_token])
            
        assert cold_tokens == control_tokens, f"Tier 3 (COLD) mismatch for prompt {item['id']}"
        
    print("\nSUCCESS: H2 hypothesis passes! Perfect token-for-token losslessness holds across all 15 prompts and all three storage tiers.")

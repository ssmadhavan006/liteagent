import json
import os
import sys
from liteagent.router.router import route_task

def run_validation():
    dataset_path = "datasets/router_validation/validation_prompts.json"
    config_path = "config/router_config.yaml"
    
    if not os.path.exists(dataset_path):
        print(f"Error: Validation dataset not found at {dataset_path}")
        sys.exit(1)
        
    with open(dataset_path, "r") as f:
        prompts = json.load(f)
        
    total = len(prompts)
    matches = 0
    mismatches = []
    
    # We clear the log file before running to start fresh
    log_dir = "experiments"
    log_file = os.path.join(log_dir, "routing_decisions.jsonl")
    if os.path.exists(log_file):
        os.remove(log_file)
        
    print(f"Running sanity validation on {total} labeled prompts...")
    print("-" * 60)
    
    for item in prompts:
        task = {
            "prompt": item["prompt"],
            "benchmark": item["benchmark"],
            "metadata": {"original_id": item["id"]}
        }
        
        result = route_task(task, config_path=config_path, log_dir=log_dir)
        pred_tier = result["model_tier"]
        exp_tier = item["expected_tier"]
        
        tier_mapping = {"Low": "Small", "Medium": "Medium", "High": "Large"}
        mapped_exp_tier = tier_mapping.get(exp_tier, exp_tier)
        
        if pred_tier == mapped_exp_tier:
            matches += 1
        else:
            mismatches.append({
                "id": item["id"],
                "benchmark": item["benchmark"],
                "prompt_snippet": item["prompt"][:60] + "...",
                "expected": mapped_exp_tier,
                "predicted": pred_tier,
                "score": round(result["metadata"]["score"], 4),
                "margin": round(result["metadata"]["routing_margin"], 4),
                "reasons": result["metadata"]["decision_reasons"]
            })
            
    accuracy = (matches / total) * 100.0
    print(f"Validation Complete.")
    print(f"Total Prompts: {total}")
    print(f"Correctly Routed: {matches}")
    print(f"Mismatched: {len(mismatches)}")
    print(f"Accuracy: {accuracy:.2f}%")
    print("-" * 60)
    
    if mismatches:
        print("Mismatch Details & Error Analysis:")
        for m in mismatches:
            print(f"ID {m['id']} [{m['benchmark']}]: Expected {m['expected']} | Predicted {m['predicted']}")
            print(f"  Snippet: {m['prompt_snippet']}")
            print(f"  Score: {m['score']} | Margin: {m['margin']} | Reasons: {m['reasons']}")
            print()
    else:
        print("All prompts routed perfectly to their expected tiers!")
        
    return accuracy

if __name__ == "__main__":
    run_validation()

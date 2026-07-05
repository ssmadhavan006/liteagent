import os
import json
import random
from collections import defaultdict

def stratify_gsm8k(input_path: str, output_path: str, target_size: int = 150, seed: int = 42):
    """
    GSM8K Stratified Sampling.
    Strata:
      - Prompt length: Short (< 150 chars) vs Long (>= 150 chars)
      - Digit count: Low (< 5 digits) vs High (>= 5 digits)
    """
    random_gen = random.Random(seed)
    items = []
    with open(input_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            item = json.loads(line)
            item["dataset_index"] = idx
            items.append(item)
            
    strata = defaultdict(list)
    for item in items:
        q = item.get("question", "")
        len_bucket = "short" if len(q) < 150 else "long"
        digit_count = sum(1 for c in q if c.isdigit())
        digit_bucket = "low" if digit_count < 5 else "high"
        strata[(len_bucket, digit_bucket)].append(item)
        
    sampled_subset = []
    # Determine proportional allocation
    total_source = len(items)
    for bucket_key, bucket_items in strata.items():
        ratio = len(bucket_items) / total_source
        bucket_target = max(1, round(ratio * target_size))
        # Sample
        shuffled = list(bucket_items)
        random_gen.shuffle(shuffled)
        sampled_subset.extend(shuffled[:bucket_target])
        
    # Trim or pad to hit target_size exactly
    if len(sampled_subset) < target_size:
        remaining = [it for it in items if it not in sampled_subset]
        random_gen.shuffle(remaining)
        sampled_subset.extend(remaining[:target_size - len(sampled_subset)])
    elif len(sampled_subset) > target_size:
        random_gen.shuffle(sampled_subset)
        sampled_subset = sampled_subset[:target_size]
        
    sampled_subset.sort(key=lambda x: x["dataset_index"])
    
    # Save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in sampled_subset:
            f.write(json.dumps(item) + "\n")
            
    print(f"GSM8K Stratified Sample: {len(sampled_subset)} items saved to {output_path}")
    return sampled_subset

def stratify_hotpotqa(input_path: str, output_path: str, target_size: int = 150, seed: int = 42):
    """
    HotpotQA Stratified Sampling.
    Strata:
      - Question length: Short (< 60 chars) vs Long (>= 60 chars)
      - Context size (words): Short (< 600 words) vs Long (>= 600 words)
    """
    random_gen = random.Random(seed)
    with open(input_path, "r", encoding="utf-8") as f:
        items = json.load(f)
        
    for idx, item in enumerate(items):
        item["dataset_index"] = idx
        
    strata = defaultdict(list)
    for item in items:
        q = item.get("question", "")
        len_bucket = "short" if len(q) < 60 else "long"
        
        # Calculate total words in all context paragraphs
        context = item.get("context", [])
        total_words = 0
        for title, sentences in context:
            total_words += sum(len(sent.split()) for sent in sentences)
            
        word_bucket = "short" if total_words < 600 else "long"
        strata[(len_bucket, word_bucket)].append(item)
        
    sampled_subset = []
    total_source = len(items)
    for bucket_key, bucket_items in strata.items():
        ratio = len(bucket_items) / total_source
        bucket_target = max(1, round(ratio * target_size))
        shuffled = list(bucket_items)
        random_gen.shuffle(shuffled)
        sampled_subset.extend(shuffled[:bucket_target])
        
    if len(sampled_subset) < target_size:
        remaining = [it for it in items if it not in sampled_subset]
        random_gen.shuffle(remaining)
        sampled_subset.extend(remaining[:target_size - len(sampled_subset)])
    elif len(sampled_subset) > target_size:
        random_gen.shuffle(sampled_subset)
        sampled_subset = sampled_subset[:target_size]
        
    sampled_subset.sort(key=lambda x: x["dataset_index"])
    
    # Save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sampled_subset, f, indent=2)
        
    print(f"HotpotQA Stratified Sample: {len(sampled_subset)} items saved to {output_path}")
    return sampled_subset

def stratify_humaneval(input_path: str, output_path: str, target_size: int = 80, seed: int = 42):
    """
    HumanEval Stratified Sampling.
    Strata:
      - Prompt length: Short (< 300 chars) vs Med (300-600 chars) vs Long (>= 600 chars)
    """
    random_gen = random.Random(seed)
    items = []
    with open(input_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            item = json.loads(line)
            item["dataset_index"] = idx
            items.append(item)
            
    strata = defaultdict(list)
    for item in items:
        p = item.get("prompt", "")
        if len(p) < 300:
            bucket = "short"
        elif len(p) < 600:
            bucket = "med"
        else:
            bucket = "long"
        strata[bucket].append(item)
        
    sampled_subset = []
    total_source = len(items)
    for bucket_key, bucket_items in strata.items():
        ratio = len(bucket_items) / total_source
        bucket_target = max(1, round(ratio * target_size))
        shuffled = list(bucket_items)
        random_gen.shuffle(shuffled)
        sampled_subset.extend(shuffled[:bucket_target])
        
    if len(sampled_subset) < target_size:
        remaining = [it for it in items if it not in sampled_subset]
        random_gen.shuffle(remaining)
        sampled_subset.extend(remaining[:target_size - len(sampled_subset)])
    elif len(sampled_subset) > target_size:
        random_gen.shuffle(sampled_subset)
        sampled_subset = sampled_subset[:target_size]
        
    sampled_subset.sort(key=lambda x: x["dataset_index"])
    
    # Save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in sampled_subset:
            f.write(json.dumps(item) + "\n")
            
    print(f"HumanEval Stratified Sample: {len(sampled_subset)} items saved to {output_path}")
    return sampled_subset

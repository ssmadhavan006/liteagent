import datetime
import hashlib
import json
import os
import time
from liteagent.router.classifier import ComplexityScorer
from liteagent.router.features import extract_raw_features, normalize_features
from liteagent.router.pruning import map_tier_and_pruning

import threading

_ROUTER_LOG_LOCK = threading.Lock()

def compute_prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

def get_decision_reasons(normalized_features: dict[str, float]) -> list[str]:
    reasons = []
    if normalized_features.get("length_chars", 0.0) > 0.4:
        reasons.append("long_prompt")
    if normalized_features.get("code_syntax_count", 0.0) > 0.4:
        reasons.append("contains_code")
    if normalized_features.get("math_operator_density", 0.0) > 0.4 or normalized_features.get("math_keyword_count", 0.0) > 0.4:
        reasons.append("math_operators")
    if normalized_features.get("logical_count", 0.0) > 0.4:
        reasons.append("logical_density")
    if normalized_features.get("question_count", 0.0) > 0.4:
        reasons.append("multiple_questions")
    if normalized_features.get("entity_density", 0.0) > 0.4:
        reasons.append("named_entities")
    return reasons

def compute_routing_margin(score: float, theta_low: float, theta_high: float, tier: str) -> float:
    if tier == "Small":
        return float(theta_low - score)
    elif tier == "Medium":
        return float(min(score - theta_low, theta_high - score))
    else:
        return float(score - theta_high)

def route_task(task: dict, config_path: str = None, log_dir: str = "experiments") -> dict:
    """
    Main entrypoint for LiteAgent routing: route_task(task)
    Returns:
    {
        "model_tier": "Small" | "Medium" | "Large",
        "execution_location": "Edge" | "Workstation",
        "active_agents": [...],
        "pruned_agents": [...],
        "metadata": {
            "config_version": int,
            "theta": float,
            "score": float,
            "confidence": float,
            "routing_margin": float,
            "decision_reasons": [...],
            "prompt_hash": str
        }
    }
    """
    start_time = time.time()

    prompt = task.get("prompt", "")
    benchmark = task.get("benchmark", "")

    # 1. Feature extraction & normalization
    raw_feats = extract_raw_features(prompt)
    norm_feats = normalize_features(raw_feats)

    # 2. Heuristic scoring
    scorer = ComplexityScorer(config_path)
    score, confidence = scorer.score_task(norm_feats)

    # 3. Agent pruning & location mapping
    tier, location, active, pruned = map_tier_and_pruning(score, scorer.theta_low, scorer.theta_high, benchmark)

    # 4. Routing metadata
    margin = compute_routing_margin(score, scorer.theta_low, scorer.theta_high, tier)
    reasons = get_decision_reasons(norm_feats)
    prompt_hash = compute_prompt_hash(prompt)

    latency_ms = (time.time() - start_time) * 1000.0

    # 5. Structured Logging
    log_entry = {
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat() + "Z",
        "config_version": scorer.version,
        "theta_low": scorer.theta_low,
        "theta_high": scorer.theta_high,
        "score": round(score, 4),
        "confidence": round(confidence, 4),
        "routing_margin": round(margin, 4),
        "tier": tier,
        "active_agents": active,
        "prompt_hash": prompt_hash,
        "decision_reasons": reasons,
        "latency_ms": round(latency_ms, 3)
    }

    if log_dir:
        try:
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, "routing_decisions.jsonl")
            with _ROUTER_LOG_LOCK:
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_entry) + "\n")
                    f.flush()
        except Exception as e:
            import sys
            print(f"Warning: Failed to write routing log: {e}", file=sys.stderr)

    return {
        "model_tier": tier,
        "execution_location": location,
        "active_agents": active,
        "pruned_agents": pruned,
        "metadata": {
            "config_version": scorer.version,
            "theta_low": scorer.theta_low,
            "theta_high": scorer.theta_high,
            "score": score,
            "confidence": confidence,
            "routing_margin": margin,
            "decision_reasons": reasons,
            "prompt_hash": prompt_hash
        }
    }

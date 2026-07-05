import math
import os
import yaml

class ComplexityScorer:
    def __init__(self, config_path: str = None):
        self.version = 1
        self.weights = {
            "length_chars": 0.35,
            "code_syntax_count": 0.40,
            "math_operator_density": 0.15,
            "math_keyword_count": 0.10,
            "logical_count": 0.10,
            "question_count": 0.05,
            "entity_density": 0.05
        }
        self.bias = -0.25
        self.theta = 0.50
        
        if config_path:
            self.load_config(config_path)

    def load_config(self, config_path: str):
        """
        Loads versioned configuration parameters from YAML.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
            
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            
        self.version = config.get("version", 1)
        self.weights = config.get("weights", self.weights)
        self.bias = config.get("bias", self.bias)
        
        routing_section = config.get("routing", {})
        self.theta = routing_section.get("theta", self.theta)

    def score_task(self, normalized_features: dict[str, float]) -> tuple[float, float]:
        """
        Computes score Sc = sigmoid(W * X + b) and confidence = 2.0 * |Sc - 0.5|
        """
        dot_product = 0.0
        for feature_name, weight in self.weights.items():
            value = normalized_features.get(feature_name, 0.0)
            dot_product += value * weight
            
        raw_score = dot_product + self.bias
        
        # Sigmoid function
        try:
            score = 1.0 / (1.0 + math.exp(-raw_score))
        except OverflowError:
            score = 0.0 if raw_score < 0.0 else 1.0
            
        confidence = 2.0 * abs(score - 0.5)
        
        return score, confidence

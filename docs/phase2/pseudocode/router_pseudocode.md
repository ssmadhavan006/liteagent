# LiteAgent — Router Pseudocode

This document contains the algorithmic pseudocode for the Complexity Router.

```python
# Complexity Router Algorithm

class ComplexityRouter:
    def __init__(self, weights: list[float], bias: float):
        # Pre-trained weights for: length, code_heuristics, math_heuristics, logical_heuristics
        self.weights = weights
        self.bias = bias

    def extract_features(self, prompt: str) -> list[float]:
        # Context length in characters
        x_len = float(len(prompt))
        
        # Heuristics counting
        code_keywords = ["def ", "import ", "class ", "fn ", "let ", "return", "{", "}", "const "]
        math_operators = ["+", "-", "*", "/", "^", "=", "%"]
        math_words = ["sum", "total", "average", "ratio", "percentage", "divide", "multiply"]
        logical_words = ["because", "therefore", "however", "since", "consequently"]
        
        x_code = sum(float(prompt.count(kw)) for kw in code_keywords)
        x_math = sum(float(prompt.count(op)) for op in math_operators) + sum(float(prompt.count(wd)) for wd in math_words)
        x_logical = sum(float(prompt.count(wd)) for wd in logical_words)
        
        # Return feature vector
        return [x_len, x_code, x_math, x_logical]

    def compute_complexity_score(self, features: list[float]) -> float:
        # Compute dot product
        raw_score = sum(f * w for f, w in zip(features, self.weights)) + self.bias
        # Sigmoid activation
        score = 1.0 / (1.0 + exp(-raw_score))
        return score

    def route_task(self, prompt: str, Theta: float) -> tuple[str, str, list[str]]:
        """
        Routes the task and returns (model_tier, execution_location, pruned_agents)
        """
        features = self.extract_features(prompt)
        S_c = self.compute_complexity_score(features)
        
        # Map master Theta to thresholds
        theta_low = 0.5 * Theta
        theta_high = 0.5 + (0.5 * Theta)
        
        if S_c < theta_low:
            # Low Complexity
            model_tier = "Small"
            location = "Edge (Pi 5)"
            pruned_agents = ["Planner", "Critic"]
        elif theta_low <= S_c < theta_high:
            # Medium Complexity
            model_tier = "Medium"
            location = "Edge (Pi 5)"
            pruned_agents = ["Critic"]
        else:
            # High Complexity
            model_tier = "Large"
            location = "Workstation"
            pruned_agents = []
            
        return model_tier, location, pruned_agents
```

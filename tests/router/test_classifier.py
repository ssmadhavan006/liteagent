import os
import math
from liteagent.router.classifier import ComplexityScorer

def test_scorer_default_initialization():
    scorer = ComplexityScorer()
    assert scorer.version == 1
    assert scorer.bias == -0.25
    assert scorer.weights["length_chars"] == 0.35
    assert scorer.theta_low == 0.25
    assert scorer.theta_high == 0.75

def test_scorer_load_config():
    # Write a temporary config
    config_path = "tests/router/temp_config.yaml"
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as f:
        f.write("""version: 2
weights:
  length_chars: 0.10
  code_syntax_count: 0.20
bias: -0.10
routing:
  theta_low: 0.30
  theta_high: 0.80
""")
    
    scorer = ComplexityScorer(config_path)
    assert scorer.version == 2
    assert scorer.bias == -0.10
    assert scorer.weights["length_chars"] == 0.10
    assert scorer.weights["code_syntax_count"] == 0.20
    assert scorer.theta_low == 0.30
    assert scorer.theta_high == 0.80
    
    # Cleanup
    if os.path.exists(config_path):
        os.remove(config_path)

def test_score_task_hand_computed():
    scorer = ComplexityScorer()
    # If all normalized features are 0.0
    # dot_product = 0.0
    # raw_score = dot_product + bias = -0.25
    # score = 1.0 / (1.0 + exp(0.25)) = 1.0 / (1.0 + 1.2840254) = 0.437823
    # confidence = 2.0 * |0.437823 - 0.5| = 2.0 * 0.062177 = 0.12435
    features = {
        "length_chars": 0.0,
        "code_syntax_count": 0.0,
        "math_operator_density": 0.0,
        "math_keyword_count": 0.0,
        "logical_count": 0.0,
        "question_count": 0.0,
        "entity_density": 0.0
    }
    score, confidence = scorer.score_task(features)
    
    expected_score = 1.0 / (1.0 + math.exp(0.25))
    expected_confidence = 2.0 * abs(expected_score - 0.5)
    
    assert abs(score - expected_score) < 1e-6
    assert abs(confidence - expected_confidence) < 1e-6

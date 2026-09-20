from liteagent.router.features import extract_raw_features, normalize_features, get_entity_density

def test_get_entity_density():
    # "What is the capital of Japan?"
    # Capitalized: "What" (first word, ignored), "Japan" (index 5, preceded by "of", count as entity).
    # Word count: 6. Expected density: 1/6 = 0.1666...
    prompt = "What is the capital of Japan?"
    assert abs(get_entity_density(prompt) - 0.1666666) < 1e-4

    # First word starts with capital, others do not
    prompt2 = "Hello world check"
    assert get_entity_density(prompt2) == 0.0

def test_extract_raw_features():
    # Prompt: "Compute 15 + 23 - 4."
    # len = 20
    # math_operators: '+' and '-' (2) -> math_operator_density = 2/20 = 0.1
    # question_count: 0
    # code_syntax_count: 0
    prompt = "Compute 15 + 23 - 4."
    features = extract_raw_features(prompt)

    assert features["length_chars"] == 20.0
    assert features["code_syntax_count"] == 0.0
    assert features["math_operator_density"] == 0.1
    assert features["math_keyword_count"] == 0.0
    assert features["logical_count"] == 0.0
    assert features["question_count"] == 0.0
    assert features["entity_density"] == 0.0

def test_extract_raw_features_code():
    # Prompt: "def add(a, b):\n    return a + b"
    # len = 32
    # code_syntax: "def " (1), "return" (1) -> total 2
    # math_operators: '+' (1) -> 1/32 = 0.03125
    prompt = "def add(a, b):\n    return a + b"
    features = extract_raw_features(prompt)

    assert features["length_chars"] == 31.0
    assert features["code_syntax_count"] == 2.0
    assert features["math_operator_density"] == 1.0 / 31.0
    assert features["entity_density"] == 0.0

def test_normalize_features():
    raw = {
        "length_chars": 8000.0,        # exceeds 4000
        "code_syntax_count": 15.0,     # exceeds 10
        "math_operator_density": 0.5,
        "math_keyword_count": 6.0,     # exceeds 5
        "logical_count": 2.0,          # 2/5 = 0.4
        "question_count": 1.0,         # 1/3 = 0.3333
        "entity_density": 0.2
    }
    norm = normalize_features(raw)

    assert norm["length_chars"] == 1.0
    assert norm["code_syntax_count"] == 1.0
    assert norm["math_operator_density"] == 0.5
    assert norm["math_keyword_count"] == 1.0
    assert norm["logical_count"] == 0.4
    assert abs(norm["question_count"] - 0.333333) < 1e-4
    assert norm["entity_density"] == 0.2

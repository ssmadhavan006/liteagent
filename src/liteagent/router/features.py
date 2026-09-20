import math

def get_entity_density(prompt: str) -> float:
    """
    Computes a lightweight heuristic for named entity density:
    The ratio of capitalized words that do not start a sentence to total words.
    """
    words = prompt.split()
    if not words:
        return 0.0

    entity_count = 0
    for i, word in enumerate(words):
        # Remove surrounding punctuation
        clean_word = word.strip(".,!?;:()[]{}'\"")
        if not clean_word or not clean_word[0].isupper() or not clean_word.isalpha():
            continue

        # Avoid counting words at the start of a sentence or first word of prompt
        if i == 0:
            continue
        prev_word = words[i-1]
        if prev_word.endswith((".", "!", "?")):
            continue

        entity_count += 1

    return entity_count / len(words)

def extract_raw_features(prompt: str) -> dict[str, float]:
    """
    Extracts raw numerical features from prompt text.
    """
    length_chars = float(len(prompt))

    # Keyword matches
    code_keywords = ["def ", "import ", "class ", "fn ", "let ", "return", "{", "}", "const "]
    code_syntax_count = sum(float(prompt.count(kw)) for kw in code_keywords)

    # Math operators and keyword counters
    math_operators = ["+", "-", "*", "/", "^", "=", "%"]
    math_op_count = sum(float(prompt.count(op)) for op in math_operators)
    math_operator_density = math_op_count / length_chars if length_chars > 0.0 else 0.0

    math_words = ["sum", "total", "average", "ratio", "percentage", "divide", "multiply"]
    math_keyword_count = sum(float(prompt.lower().count(wd)) for wd in math_words)

    # Logical keywords
    logical_words = ["because", "therefore", "however", "since", "consequently"]
    logical_count = sum(float(prompt.lower().count(wd)) for wd in logical_words)

    question_count = float(prompt.count("?"))
    entity_density = get_entity_density(prompt)

    return {
        "length_chars": length_chars,
        "code_syntax_count": code_syntax_count,
        "math_operator_density": math_operator_density,
        "math_keyword_count": math_keyword_count,
        "logical_count": logical_count,
        "question_count": question_count,
        "entity_density": entity_density
    }

def normalize_features(raw_features: dict[str, float]) -> dict[str, float]:
    """
    Normalizes extracted feature values to the range [0.0, 1.0].
    """
    length_val = raw_features["length_chars"]
    log_len = min(math.log(length_val) / math.log(4000.0), 1.0) if length_val > 1.0 else 0.0
    return {
        "length_chars": log_len,
        "code_syntax_count": min(raw_features["code_syntax_count"] / 10.0, 1.0),
        "math_operator_density": min(raw_features["math_operator_density"], 1.0),
        "math_keyword_count": min(raw_features["math_keyword_count"] / 5.0, 1.0),
        "logical_count": min(raw_features["logical_count"] / 5.0, 1.0),
        "question_count": min(raw_features["question_count"] / 3.0, 1.0),
        "entity_density": min(raw_features["entity_density"], 1.0)
    }

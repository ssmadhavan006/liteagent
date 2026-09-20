import re
from liteagent.eval.sandbox import run_sandboxed_code

def extract_code_from_markdown(text: str) -> str:
    """
    Extracts executable Python code from markdown fenced blocks if present.
    """
    pattern = r"```(?:python)?\s*\n(.*?)\n```"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1)
    return text.strip()

def score_humaneval(prediction: str, test_code: str, timeout: float = 3.0) -> dict:
    """
    Computes single-sample pass@1 (1 generation per task) for HumanEval.
    Returns score (1.0 for success, 0.0 for failure) and sandbox diagnostic metrics.
    """
    clean_prediction = extract_code_from_markdown(prediction)
    res = run_sandboxed_code(clean_prediction, test_code, timeout=timeout)
    score = 1.0 if res["success"] else 0.0
    return {
        "pass_status": score,
        "success": res["success"],
        "timeout_occurred": res["timeout_occurred"],
        "failure_category": res["failure_category"],
        "stderr": res["stderr"]
    }

from liteagent.eval.sandbox import run_sandboxed_code

def score_humaneval(prediction: str, test_code: str, timeout: float = 3.0) -> dict:
    """
    Computes single-sample pass@1 (1 generation per task) for HumanEval.
    Returns score (1.0 for success, 0.0 for failure) and sandbox diagnostic metrics.
    """
    res = run_sandboxed_code(prediction, test_code, timeout=timeout)
    score = 1.0 if res["success"] else 0.0
    return {
        "pass_status": score,
        "success": res["success"],
        "timeout_occurred": res["timeout_occurred"],
        "failure_category": res["failure_category"],
        "stderr": res["stderr"]
    }

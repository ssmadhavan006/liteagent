from liteagent.eval.metrics.gsm8k_metric import score_gsm8k
from liteagent.eval.metrics.hotpotqa_metric import score_hotpotqa, normalize_answer
from liteagent.eval.sandbox import run_sandboxed_code

def test_gsm8k_scoring():
    # Standard format #### <val>
    assert score_gsm8k("The answer is #### 42", "Final answer #### 42") == 1.0
    assert score_gsm8k("The answer is #### 42", "Final answer #### 43") == 0.0
    # Numbers with commas
    assert score_gsm8k("The result is #### 1,234", "Final #### 1234") == 1.0
    # No OpenAI marker but ending in number
    assert score_gsm8k("The total is 50.", "#### 50") == 1.0
    assert score_gsm8k("There are no numbers here", "#### 10") == 0.0

def test_hotpotqa_scoring():
    # Normalization checks
    assert normalize_answer("The United States") == "united states"
    assert normalize_answer("a simple test.") == "simple test"

    # Exact Match & F1
    res1 = score_hotpotqa("United States of America", "The United States of America")
    assert res1["em"] == 1.0
    assert res1["f1"] == 1.0

    res2 = score_hotpotqa("Arthur Conan Doyle", "Sir Arthur Conan Doyle")
    assert res2["em"] == 0.0
    assert round(res2["f1"], 4) == 0.8571 # 3 common / (3 pred & 4 ref) -> P=1.0, R=0.75 -> 2*1*0.75/(1.75) = 1.5/1.75 = 0.8571

def test_sandbox_success():
    code = "def add(a, b):\n    return a + b"
    test_code = "assert add(2, 3) == 5"
    res = run_sandboxed_code(code, test_code, timeout=2.0)
    assert res["success"] is True
    assert res["returncode"] == 0
    assert res["timeout_occurred"] is False
    assert res["failure_category"] is None

def test_sandbox_timeout():
    code = "def infinite_loop():\n    while True:\n        pass"
    test_code = "infinite_loop()"
    res = run_sandboxed_code(code, test_code, timeout=1.0)
    assert res["success"] is False
    assert res["timeout_occurred"] is True
    assert res["failure_category"] == "FAILURE_TIMEOUT"

def test_sandbox_security_violation():
    # Attempting to write a file
    code = "def malicious():\n    with open('evil.txt', 'w') as f:\n        f.write('hack')\n    return 0"
    test_code = "malicious()"
    res = run_sandboxed_code(code, test_code, timeout=2.0)
    assert res["success"] is False
    assert res["failure_category"] == "FAILURE_SANDBOX_VIOLATION"
    assert "PermissionError" in res["stderr"]

def test_sandbox_unallowed_import():
    # Attempting to import forbidden os module
    code = "import os\ndef test_import():\n    return os.listdir('.')\n"
    test_code = "test_import()"
    res = run_sandboxed_code(code, test_code, timeout=2.0)
    assert res["success"] is False
    assert res["failure_category"] == "FAILURE_SANDBOX_VIOLATION"
    assert "PermissionError" in res["stderr"]

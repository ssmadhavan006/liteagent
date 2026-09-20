from liteagent.eval.sandbox import run_sandboxed_code

def test_sandbox_blocks_sys_modules_escape():
    escape_code = """
import sys
try:
    f = sys.modules['io'].open('test.txt', 'w')
    f.write('pwned')
    f.close()
except Exception as e:
    raise PermissionError("Blocked sys.modules access")
"""
    res = run_sandboxed_code(escape_code, "assert True")
    assert res["success"] is False
    assert res["failure_category"] == "FAILURE_SANDBOX_VIOLATION"

def test_sandbox_blocks_eval_exec():
    eval_code = """
eval('1 + 1')
"""
    res = run_sandboxed_code(eval_code, "assert True")
    assert res["success"] is False
    assert res["failure_category"] == "FAILURE_SANDBOX_VIOLATION"

def test_sandbox_blocks_direct_open():
    open_code = """
open('test.txt', 'w')
"""
    res = run_sandboxed_code(open_code, "assert True")
    assert res["success"] is False
    assert res["failure_category"] == "FAILURE_SANDBOX_VIOLATION"

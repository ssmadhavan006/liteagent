"""
Tests for execution-based verification.

The load-bearing property is that a *correct* program is not reported as wrong.
The LLM Critic failed precisely there (specificity 0.09 on HumanEval: it
rejected almost everything, which still scores well on sensitivity alone).
"""

import pytest

from liteagent.eval.exec_verifier import (
    extract_examples,
    verify_completion,
    verify_program,
)

PROMPT = '''from typing import List


def add_one(xs: List[int]) -> List[int]:
    """Add one to each element.
    >>> add_one([1, 2])
    [2, 3]
    >>> add_one([])
    []
    """
'''

ARROW_PROMPT = '''
def double(n: int) -> int:
    """Double it.
    For example:
    double(2) == 4
    double(5) ==> 10
    """
'''


def test_extracts_doctest_examples():
    ex = extract_examples(PROMPT, "add_one")
    assert ("add_one([1, 2])", "[2, 3]") in ex
    assert ("add_one([])", "[]") in ex


def test_extracts_arrow_and_equality_forms():
    ex = extract_examples(ARROW_PROMPT, "double")
    calls = dict(ex)
    assert calls.get("double(2)") == "4"
    assert calls.get("double(5)") == "10"


def test_rejects_calls_with_non_literal_arguments():
    """`is_happy(a)` in a docstring means the string "a"; executing it raises."""
    prompt = '''
def is_happy(s):
    """Check.
    is_happy(a) => False
    is_happy(abcd) => True
    """
'''
    assert extract_examples(prompt, "is_happy") == []


def test_correct_completion_passes():
    res = verify_completion("    return [x + 1 for x in xs]", PROMPT, "add_one")
    assert res["verdict"] == "pass"
    assert res["examples_run"] == 2


def test_wrong_completion_fails():
    res = verify_completion("    return [x - 1 for x in xs]", PROMPT, "add_one")
    assert res["verdict"] == "fail"


def test_syntax_error_fails_without_execution():
    res = verify_completion("    return [x + for x in xs]", PROMPT, "add_one")
    assert res["verdict"] == "fail"
    assert "syntax" in res["reason"].lower()
    assert res["examples_run"] == 0


def test_empty_completion_fails():
    assert verify_completion("", PROMPT, "add_one")["verdict"] == "fail"


def test_missing_examples_are_unverifiable_not_pass():
    """
    Treating "no examples" as "correct" is how a verifier silently approves
    everything, which is the Critic's failure mode.
    """
    prompt = 'def f(x):\n    """No examples here."""\n'
    res = verify_completion("    return x", prompt, "f")
    assert res["verdict"] == "unverifiable"
    assert res["verdict"] != "pass"


def test_float_results_compared_with_tolerance():
    prompt = '''
def half(n: float) -> float:
    """Halve it.
    >>> half(5)
    2.5
    """
'''
    assert verify_completion("    return n / 2", prompt, "half")["verdict"] == "pass"


def test_verifier_never_receives_the_hidden_test_suite():
    """
    Escalating on the benchmark's own tests would consult the answer key at
    inference time. verify_program's inputs are the program and prompt-derived
    examples only.
    """
    import inspect
    src = inspect.getsource(verify_completion) + inspect.getsource(verify_program)
    assert '"test"' not in src and "['test']" not in src


@pytest.mark.slow
def test_reference_solutions_are_not_rejected():
    import json
    import os
    path = os.path.join("data", "humaneval", "subset.jsonl")
    if not os.path.exists(path):
        pytest.skip("humaneval subset unavailable")

    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    decided = rejected = 0
    for r in rows:
        res = verify_completion(r["canonical_solution"], r["prompt"], r["entry_point"])
        if res["verdict"] == "pass":
            decided += 1
        elif res["verdict"] == "fail":
            decided += 1
            rejected += 1

    # One known false rejection: HumanEval's `median` docstring states an
    # incorrect expected value (claims 15.0 where the answer is 8.0), so a
    # prompt-derived check cannot pass it.
    assert rejected <= 1, f"{rejected} reference solutions rejected"
    assert decided >= 50

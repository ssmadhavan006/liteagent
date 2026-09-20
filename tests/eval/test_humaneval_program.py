"""
Guards HumanEval program assembly.

A HumanEval prompt is a signature plus docstring and the model supplies the
body, so the executed program must be prompt + completion. Scoring the
completion alone produced a non-runnable fragment and returned 0 for every
task, including the dataset's own reference solutions.
"""

import json
import os

import pytest

from liteagent.eval.metrics.humaneval_metric import build_program, score_humaneval

PROMPT = 'from typing import List\n\n\ndef add_one(xs: List[int]) -> List[int]:\n    """Add one."""\n'
TEST = "assert add_one([1, 2]) == [2, 3]"


def test_completion_style_is_appended_to_prompt():
    completion = "    return [x + 1 for x in xs]"
    program = build_program(completion, prompt=PROMPT, entry_point="add_one")
    assert program.startswith("from typing import List")
    assert "def add_one" in program
    assert program.count("def add_one") == 1


def test_body_indentation_is_preserved():
    """Stripping unindents only the first line and breaks every later one."""
    completion = "    result = []\n    for x in xs:\n        result.append(x + 1)\n    return result"
    program = build_program(completion, prompt=PROMPT, entry_point="add_one")
    assert "\n    result = []" in program
    compile(program, "<program>", "exec")


def test_flush_left_body_is_reindented():
    completion = "return [x + 1 for x in xs]"
    program = build_program(completion, prompt=PROMPT, entry_point="add_one")
    compile(program, "<program>", "exec")


def test_restated_function_is_not_duplicated():
    """Instruction-tuned models often rewrite the whole function."""
    completion = "def add_one(xs):\n    return [x + 1 for x in xs]"
    program = build_program(completion, prompt=PROMPT, entry_point="add_one")
    assert program.count("def add_one") == 1
    assert "from typing import List" in program, "prompt imports must be retained"
    compile(program, "<program>", "exec")


def test_markdown_fences_are_stripped():
    completion = "```python\ndef add_one(xs):\n    return [x + 1 for x in xs]\n```"
    program = build_program(completion, prompt=PROMPT, entry_point="add_one")
    assert "```" not in program
    compile(program, "<program>", "exec")


def test_end_to_end_completion_scores_one():
    res = score_humaneval(
        "    return [x + 1 for x in xs]",
        TEST,
        prompt=PROMPT,
        entry_point="add_one",
    )
    assert res["pass_status"] == 1.0, res["stderr"]


def test_assertions_actually_execute():
    """
    HumanEval's `test` field only defines check(candidate); calling it is the
    harness's job. Without the call the assertions never run, the script exits
    0, and every syntactically valid program scores as correct.
    """
    from liteagent.eval.metrics.humaneval_metric import build_test_harness

    test_code = "def check(candidate):\n    assert candidate([1]) == [2]\n"
    assert "check(add_one)" in build_test_harness(test_code, "add_one")
    # An existing call must not be duplicated.
    already = test_code + "\ncheck(add_one)\n"
    assert build_test_harness(already, "add_one").count("check(add_one)") == 1


def test_program_that_compiles_but_is_wrong_does_not_pass():
    """The regression that made every HumanEval task score 1.0."""
    res = score_humaneval(
        "    return []",  # compiles, runs, and is wrong
        "def check(candidate):\n    assert candidate([1, 2]) == [2, 3]\n",
        prompt=PROMPT,
        entry_point="add_one",
    )
    assert res["pass_status"] == 0.0, "a compiling but incorrect program must fail"


def test_wrong_completion_scores_zero():
    res = score_humaneval(
        "    return [x - 1 for x in xs]",
        TEST,
        prompt=PROMPT,
        entry_point="add_one",
    )
    assert res["pass_status"] == 0.0
    assert res["failure_category"] == "FAILURE_ASSERTION"


@pytest.mark.slow
def test_reference_solutions_all_pass():
    """
    The dataset's own solutions must score 1.0. If they do not, the metric is
    measuring the harness rather than the model.
    """
    path = os.path.join("data", "humaneval", "subset.jsonl")
    if not os.path.exists(path):
        pytest.skip("humaneval subset unavailable")

    from liteagent.eval.sandbox import UNSCOREABLE_TASKS

    items = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()][:15]
    failures = []
    for item in items:
        if item["entry_point"] in UNSCOREABLE_TASKS:
            continue
        res = score_humaneval(
            item["canonical_solution"], item["test"],
            prompt=item["prompt"], entry_point=item["entry_point"],
        )
        if res["pass_status"] != 1.0:
            failures.append((item["entry_point"], res["failure_category"]))

    assert not failures, f"reference solutions failed: {failures}"

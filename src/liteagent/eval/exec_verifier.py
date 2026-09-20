"""
Execution-based verification for code tasks.

The LLM Critic cannot discriminate correct from incorrect answers (Youden's J
near zero, see docs/phase9/router_capability_analysis.md §8), which makes
verification-gated escalation unviable. For code, though, correctness is
partially *decidable*: the program can simply be run.

**This verifier never touches the benchmark's hidden test suite.** Deciding
escalation from `item["test"]` would consult the answer key at inference time
and inflate every downstream number. It uses only the worked examples embedded
in the prompt's docstring — exactly the information the model itself was given.

Verdicts:

    pass          compiles and every extracted example matches
    fail          does not compile, or an example disagrees
    unverifiable  compiles, but no example could be extracted

`unverifiable` is deliberately distinct from `pass`. Roughly a fifth of
HumanEval prompts state their examples in prose that cannot be executed, and
silently treating those as correct would recreate the Critic's failure mode of
approving everything.
"""

import ast
import re

from liteagent.eval.sandbox import run_sandboxed_code

# Comparison helper injected into the verification script. Float results need a
# tolerance; exact equality on floats would report correct solutions as wrong.
_EQ_HELPER = '''
def _eq(a, b):
    if isinstance(a, float) or isinstance(b, float):
        try:
            return abs(a - b) < 1e-6
        except TypeError:
            return a == b
    return a == b
'''


def _is_literal(text: str) -> bool:
    try:
        ast.literal_eval(text)
        return True
    except (ValueError, SyntaxError):
        return False


def _is_call(text: str, entry_point: str) -> bool:
    """
    Accepts a call on the target function whose arguments are all literals.

    The literal requirement matters: several prompts write arrow-form examples
    with unquoted strings (`is_happy(a) => False`), where `a` is an undefined
    name. Executing that raises NameError and would report a correct solution as
    wrong. Such examples are skipped rather than trusted.
    """
    if entry_point and entry_point not in text:
        return False
    try:
        node = ast.parse(text.strip(), mode="eval")
    except SyntaxError:
        return False
    if not isinstance(node.body, ast.Call):
        return False

    for arg in list(node.body.args) + [kw.value for kw in node.body.keywords]:
        try:
            ast.literal_eval(arg)
        except (ValueError, SyntaxError):
            return False
    return True


def extract_examples(prompt: str, entry_point: str) -> list[tuple[str, str]]:
    """
    Recovers (call, expected) pairs from the docstring.

    Handles the three shapes HumanEval uses: `>>>` doctests, `call == value`,
    and arrow forms (`->`, `==>`, the unicode arrow). Only pairs whose expected
    value is a Python literal are kept, since anything else cannot be compared
    without evaluating untrusted text.
    """
    examples: list[tuple[str, str]] = []
    lines = prompt.splitlines()

    # 1. Doctest blocks: ">>> call" then the expected value on following lines.
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith(">>>"):
            continue
        call = stripped[3:].strip()
        if not _is_call(call, entry_point):
            continue
        for follow in lines[i + 1:]:
            nxt = follow.strip()
            if not nxt or nxt.startswith(">>>"):
                break
            if _is_literal(nxt):
                examples.append((call, nxt))
            break

    # 2. Inline forms on a single line.
    arrow = re.compile(
        r"([A-Za-z_]\w*\s*\(.*?\))\s*(?:==>|=>|➞|->|==)\s*([^\n#]+)"
    )
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(">>>"):
            continue
        for call, expected in arrow.findall(stripped):
            call = call.strip()
            expected = expected.strip().rstrip(".,;")
            if _is_call(call, entry_point) and _is_literal(expected):
                examples.append((call, expected))

    # De-duplicate while preserving order.
    seen = set()
    unique = []
    for pair in examples:
        if pair not in seen:
            seen.add(pair)
            unique.append(pair)
    return unique


def verify_program(
    program: str,
    examples: list[tuple[str, str]],
    timeout: float = 3.0,
) -> dict:
    """Runs the program against prompt-derived examples in the sandbox."""
    if not program.strip():
        return {"verdict": "fail", "reason": "empty program", "examples_run": 0}

    try:
        compile(program, "<candidate>", "exec")
    except SyntaxError as exc:
        return {"verdict": "fail", "reason": f"syntax error: {exc.msg}", "examples_run": 0}

    if not examples:
        return {"verdict": "unverifiable", "reason": "no executable examples in prompt",
                "examples_run": 0}

    checks = "\n".join(
        f"assert _eq({call}, {expected}), {i}" for i, (call, expected) in enumerate(examples)
    )
    res = run_sandboxed_code(program, _EQ_HELPER + "\n" + checks, timeout=timeout)

    if res["success"]:
        return {"verdict": "pass", "reason": "", "examples_run": len(examples)}
    return {
        "verdict": "fail",
        "reason": res.get("failure_category") or "example mismatch",
        "examples_run": len(examples),
    }


def verify_completion(
    completion: str,
    prompt: str,
    entry_point: str,
    timeout: float = 3.0,
) -> dict:
    """Assembles the program as the metric does, then verifies it."""
    from liteagent.eval.metrics.humaneval_metric import build_program

    program = build_program(completion, prompt=prompt, entry_point=entry_point)
    result = verify_program(program, extract_examples(prompt, entry_point), timeout=timeout)
    result["program"] = program
    return result

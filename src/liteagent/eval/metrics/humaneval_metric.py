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


def _extract_preserving_indent(text: str) -> str:
    """
    Like extract_code_from_markdown, but never strips leading whitespace.

    A function body's indentation is load-bearing: stripping it unindents only
    the first line and every subsequent line then raises IndentationError.
    """
    match = re.search(r"```(?:python)?\s*\n(.*?)\n```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1)
    return text.rstrip()


def build_program(prediction: str, prompt: str = "", entry_point: str = "") -> str:
    """
    Assembles the program to execute from a model completion.

    A HumanEval prompt is a function signature plus docstring, and the model
    supplies the body. The body alone is not a runnable program, so it must be
    concatenated onto the prompt before execution.

    Instruction-tuned models often restate the whole function instead of
    continuing it. Concatenating in that case would produce a duplicate or
    misindented definition, so the completion is used as-is whenever it already
    defines the target function.
    """
    code = _extract_preserving_indent(prediction)

    if not prompt:
        return code.strip()

    defines_entry = bool(
        entry_point and re.search(rf"^\s*def\s+{re.escape(entry_point)}\s*\(", code, re.MULTILINE)
    )
    if defines_entry:
        # The model restated the signature; keep any imports from the prompt,
        # which the restated version usually drops.
        preamble = "\n".join(
            line for line in prompt.splitlines()
            if line.startswith(("import ", "from "))
        )
        code = code.strip()
        return f"{preamble}\n{code}" if preamble else code

    # Completion style: re-indent only if the model emitted a flush-left body,
    # which would otherwise fall outside the function.
    first = next((ln for ln in code.splitlines() if ln.strip()), "")
    if first and not first[0].isspace():
        code = "\n".join(("    " + ln if ln.strip() else ln) for ln in code.splitlines())

    return prompt + code


def score_humaneval(
    prediction: str,
    test_code: str,
    timeout: float = 3.0,
    prompt: str = "",
    entry_point: str = "",
) -> dict:
    """
    Computes single-sample pass@1 (1 generation per task) for HumanEval.
    Returns score (1.0 for success, 0.0 for failure) and sandbox diagnostic metrics.
    """
    program = build_program(prediction, prompt=prompt, entry_point=entry_point)
    res = run_sandboxed_code(program, test_code, timeout=timeout)
    score = 1.0 if res["success"] else 0.0
    return {
        "pass_status": score,
        "success": res["success"],
        "timeout_occurred": res["timeout_occurred"],
        "failure_category": res["failure_category"],
        "stderr": res["stderr"],
    }

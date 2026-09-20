import pytest

from liteagent.router.capability_labels import (
    TIER_ORDER,
    build_prompt,
    cohens_kappa,
    format_hotpotqa_prompt,
    is_correct,
)


def test_tier_order_runs_smallest_to_largest():
    assert [t for t, _ in TIER_ORDER] == ["Low", "Medium", "High"]
    assert [m for _, m in TIER_ORDER] == ["llama3.2:1b", "llama3.2:3b", "llama3.1:8b"]


def test_build_prompt_without_fewshot_is_bare_body_plus_contract():
    prompt = build_prompt("gsm8k", {"question": "2+2?"}, few_shot=False)
    assert prompt.endswith("2+2?")
    assert "#### <answer>" in prompt, "output contract must be present"

    with pytest.raises(ValueError):
        build_prompt("nope", {}, few_shot=False)


def test_build_prompt_applies_fewshot_exemplars():
    prompt = build_prompt("gsm8k", {"question": "2+2?"})
    assert prompt.endswith("2+2?")
    # Exemplars must precede the task and demonstrate the answer marker.
    assert prompt.count("####") >= 5
    assert "Question:" in prompt


def test_humaneval_stays_zero_shot():
    """Prepending unrelated functions degrades completion quality."""
    prompt = build_prompt("humaneval", {"prompt": "def f():"})
    assert prompt.endswith("def f():")
    assert "Question:" not in prompt


def test_labelling_protocol_matches_harness_protocol():
    """Labels are only transferable if produced under the evaluated protocol."""
    from liteagent.eval.fewshot import build_prefix
    from liteagent.eval.harness import BENCHMARK_SYSTEM_PROMPTS

    prompt = build_prompt("gsm8k", {"question": "Q?"})
    assert BENCHMARK_SYSTEM_PROMPTS["gsm8k"] in prompt
    assert build_prefix("gsm8k") in prompt


def test_hotpotqa_prompt_includes_titles_and_question():
    item = {"context": [["Inception", ["Nolan directed it."]]], "question": "Who directed it?"}
    prompt = format_hotpotqa_prompt(item)
    assert "[Inception]" in prompt
    assert "Who directed it?" in prompt


def test_gsm8k_correctness_uses_exact_match():
    item = {"answer": "The result is 5.\n#### 18"}
    solved, score = is_correct("gsm8k", "the answer is #### 18", item)
    assert solved is True and score == 1.0

    solved, _ = is_correct("gsm8k", "the answer is 17", item)
    assert solved is False


def test_hotpotqa_correctness_uses_f1_threshold():
    item = {"answer": "Christopher Nolan"}
    solved, score = is_correct("hotpotqa", "Christopher Nolan", item)
    assert solved is True and score == pytest.approx(1.0)

    solved, _ = is_correct("hotpotqa", "a baker in Vienna", item)
    assert solved is False


def test_cohens_kappa_bounds():
    assert cohens_kappa(["Low", "High"], ["Low", "High"]) == pytest.approx(1.0)
    # Perfect disagreement on a balanced two-category set is negative.
    assert cohens_kappa(["Low", "High"], ["High", "Low"]) < 0
    assert cohens_kappa([], []) == 0.0


def test_cohens_kappa_corrects_for_chance_agreement():
    # Identical constant labels: agreement is entirely chance, kappa undefined -> 1.0 guard.
    assert cohens_kappa(["Low"] * 5, ["Low"] * 5) == 1.0
    a = ["Low", "Low", "Medium", "High"]
    b = ["Low", "Medium", "Medium", "High"]
    k = cohens_kappa(a, b)
    assert 0.0 < k < 1.0

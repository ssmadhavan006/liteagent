"""
Fixed few-shot prefixes for the evaluation protocol.

Small instruction-tuned models do not reliably follow a bare format
instruction: measured on llama3.2:1b, zero-shot GSM8K generations reason
correctly but omit the answer marker, so extraction falls back to "last number
in the text" and scores correct solves as failures. Few-shot prompting is the
standard protocol for these benchmarks and is what makes the quality numbers
meaningful.

Two properties matter for validity:

1. **Identical across systems.** The harness issues the same prefix to every
   configuration, so no system gains an advantage from knowing the output
   format. This is enforced by construction: the prefix is built here and
   passed through `EvaluationHarness`, never by an individual runner.

2. **No contamination.** Exemplars are hand-authored rather than sampled from
   the benchmark splits, so no evaluated item ever appears in its own prompt.
"""

# Hand-authored, in the style of the chain-of-thought GSM8K protocol. These are
# not drawn from the GSM8K dataset.
GSM8K_EXEMPLARS = [
    (
        "A baker has 24 muffins. He sells 9 in the morning and 7 in the afternoon. "
        "How many muffins are left?",
        "He sells 9 + 7 = 16 muffins in total.\n"
        "That leaves 24 - 16 = 8 muffins.\n"
        "#### 8",
    ),
    (
        "A train travels 60 miles per hour for 3 hours, then 40 miles per hour for 2 hours. "
        "How many miles did it travel?",
        "In the first stretch it travels 60 * 3 = 180 miles.\n"
        "In the second stretch it travels 40 * 2 = 80 miles.\n"
        "In total it travels 180 + 80 = 260 miles.\n"
        "#### 260",
    ),
    (
        "Maria buys 5 notebooks for $3 each and 2 pens for $2 each. "
        "She pays with a $50 note. How much change does she get?",
        "The notebooks cost 5 * 3 = $15.\n"
        "The pens cost 2 * 2 = $4.\n"
        "Her total is 15 + 4 = $19.\n"
        "Her change is 50 - 19 = $31.\n"
        "#### 31",
    ),
    (
        "A class has 30 students. Two fifths of them play football and half of the rest play chess. "
        "How many students play chess?",
        "Two fifths of 30 is 30 * 2 / 5 = 12 students who play football.\n"
        "That leaves 30 - 12 = 18 students.\n"
        "Half of those play chess: 18 / 2 = 9.\n"
        "#### 9",
    ),
    (
        "A tank holds 120 litres. It is three quarters full and then 30 litres are drained. "
        "How many litres remain?",
        "Three quarters of 120 is 120 * 3 / 4 = 90 litres.\n"
        "After draining, 90 - 30 = 60 litres remain.\n"
        "#### 60",
    ),
]

# Short span-style answers. Deliberately context-free: HotpotQA prompts already
# carry ten paragraphs, and attaching contexts to exemplars would overflow the
# 4096-token window. These teach answer brevity, not retrieval.
HOTPOTQA_EXEMPLARS = [
    ("Which country hosted the 1988 Summer Olympics?", "South Korea"),
    ("Who wrote the novel 'Wuthering Heights'?", "Emily Bronte"),
    ("What is the chemical symbol for potassium?", "K"),
]


def gsm8k_prefix() -> str:
    blocks = [
        f"Question: {q}\nAnswer: {a}" for q, a in GSM8K_EXEMPLARS
    ]
    return "\n\n".join(blocks) + "\n\n"


def hotpotqa_prefix() -> str:
    blocks = [f"Question: {q}\nAnswer: {a}" for q, a in HOTPOTQA_EXEMPLARS]
    return "\n\n".join(blocks) + "\n\n"


# HumanEval is evaluated zero-shot by convention: the prompt is itself the
# function signature and docstring to complete, and prepending unrelated
# functions degrades completion quality.
FEWSHOT_PREFIXES = {
    "gsm8k": gsm8k_prefix,
    "hotpotqa": hotpotqa_prefix,
    "humaneval": lambda: "",
}


def build_prefix(benchmark: str, enabled: bool = True) -> str:
    """Returns the few-shot prefix for a benchmark, or '' when disabled."""
    if not enabled:
        return ""
    builder = FEWSHOT_PREFIXES.get(benchmark)
    return builder() if builder else ""


def shot_count(benchmark: str) -> int:
    """Number of exemplars, recorded in results so the protocol is auditable."""
    if benchmark == "gsm8k":
        return len(GSM8K_EXEMPLARS)
    if benchmark == "hotpotqa":
        return len(HOTPOTQA_EXEMPLARS)
    return 0

import re

def extract_gsm8k_answer(text: str) -> str:
    """
    Extracts the final numeric answer from GSM8K ground truth or model responses.
    Matches OpenAI's standard format ( #### <num> ) or extracts the last number.
    """
    # 1. Check for OpenAI marker #### <val>
    openai_match = re.search(r"####\s*(-?\d[\d,]*\.?\d*)", text)
    if openai_match:
        val = openai_match.group(1).replace(",", "")
        return val.rstrip(".")
        
    # 2. Otherwise extract the last numeric token in the text
    # We clean the text of surrounding punctuation except math signs/numbers
    clean_tokens = [w.strip(".,!?;:") for w in text.split()]
    for token in reversed(clean_tokens):
        match = re.match(r"^-?\d[\d,]*\.?\d*$", token)
        if match:
            return token.replace(",", "")
    return ""

def score_gsm8k(prediction: str, reference: str) -> float:
    """
    Returns 1.0 if the extracted predicted number matches the reference number, else 0.0.
    """
    pred_num = extract_gsm8k_answer(prediction)
    ref_num = extract_gsm8k_answer(reference)
    if not ref_num:
        return 0.0
    return 1.0 if pred_num == ref_num else 0.0

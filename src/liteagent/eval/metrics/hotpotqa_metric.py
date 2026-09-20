import re
import string

def normalize_answer(s: str) -> str:
    """
    Lowercases, removes punctuation, removes articles (a, an, the), and standardizes whitespace.
    Matches standard SQuAD/HotpotQA evaluation normalization.
    """
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def score_hotpotqa(prediction: str, reference: str) -> dict[str, float]:
    """
    Computes Exact Match (EM) and F1 score for prediction vs reference answer.
    """
    norm_pred = normalize_answer(prediction)
    norm_ref = normalize_answer(reference)

    # 1. Exact Match
    em = 1.0 if norm_pred == norm_ref else 0.0

    # 2. Token F1
    pred_tokens = norm_pred.split()
    ref_tokens = norm_ref.split()

    if not pred_tokens or not ref_tokens:
        f1 = 1.0 if norm_pred == norm_ref else 0.0
        return {"em": em, "f1": f1}

    common_tokens = []
    # Count overlaps preserving multiples
    ref_counts = {}
    for tok in ref_tokens:
        ref_counts[tok] = ref_counts.get(tok, 0) + 1

    for tok in pred_tokens:
        if tok in ref_counts and ref_counts[tok] > 0:
            common_tokens.append(tok)
            ref_counts[tok] -= 1

    num_same = len(common_tokens)
    if num_same == 0:
        return {"em": em, "f1": 0.0}

    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    f1 = 2.0 * precision * recall / (precision + recall)

    return {"em": em, "f1": f1}

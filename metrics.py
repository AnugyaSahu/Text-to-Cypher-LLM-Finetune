def exact_match(prediction: str, ground_truth: str) -> int:
    """
    Strict match — good baseline but penalizes
    semantically equivalent queries
    """
    return int(prediction.strip() == ground_truth.strip())


def token_f1(prediction: str, ground_truth: str) -> float:
    """
    Why Token F1?
    - Measures token overlap between prediction and ground truth
    - More forgiving than Exact Match, captures partial correctness
    - More meaningful than BLEU for structured query language
    - common token betwen pred_set and truth_set
    - precision = how much prediction is correct
    - recall - how much of a ground truth was captured
    - F1 = Harmonic mean
    """
    pred_tokens = prediction.strip().split()
    truth_tokens = ground_truth.strip().split()

    # edge case — if either is empty
    if not pred_tokens or not truth_tokens:
        return 0.0

    # token calculated with simple white space splitting
    # set , so no duplicates
    pred_set = set(pred_tokens)
    truth_set = set(truth_tokens)

    #overlapping tokens between prediction and ground truth
    common = pred_set & truth_set

    precision = len(common) / len(pred_set)
    recall = len(common) / len(truth_set)

    if precision + recall == 0:
        return 0.0

    f1 = 2 * (precision * recall) / (precision + recall)
    return round(f1, 4)


def compute_metrics(prediction: str, ground_truth: str) -> dict:
    return {
        "exact_match": exact_match(prediction, ground_truth),
        "token_f1": token_f1(prediction, ground_truth)
    }

from typing import List, Dict, Any
import re
import numpy as np
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text


def compute_accuracy(preds: List[str], golds: List[str]) -> float:
    matches = sum(1 for p, g in zip(preds, golds) if normalize_text(p) == normalize_text(g))
    return matches / len(golds) if golds else 0.0


def compute_precision_recall_f1(preds: List[str], golds: List[str]) -> Dict[str, float]:
    total_tp, total_fp, total_fn = 0, 0, 0

    for pred, gold in zip(preds, golds):
        p_tokens = set(normalize_text(pred).split())
        g_tokens = set(normalize_text(gold).split())

        tp = len(p_tokens & g_tokens)
        fp = len(p_tokens - g_tokens)
        fn = len(g_tokens - p_tokens)

        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0

    return {"precision": precision, "recall": recall, "f1": f1}


def compute_bleu(preds: List[str], golds: List[str]) -> float:
    smoothie = SmoothingFunction().method4
    scores = []
    for pred, gold in zip(preds, golds):
        g_tokens = [normalize_text(gold).split()]
        p_tokens = normalize_text(pred).split()
        try:
            score = sentence_bleu(g_tokens, p_tokens, smoothing_function=smoothie)
        except ZeroDivisionError:
            score = 0.0
        scores.append(score)
    return float(np.mean(scores)) if scores else 0.0


def compute_rouge(preds: List[str], golds: List[str]) -> Dict[str, float]:
    def lcs(a, b):
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m):
            for j in range(n):
                if a[i] == b[j]:
                    dp[i+1][j+1] = dp[i][j] + 1
                else:
                    dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
        return dp[m][n]

    scores = []
    for pred, gold in zip(preds, golds):
        p_tokens, g_tokens = pred.split(), gold.split()
        lcs_len = lcs(p_tokens, g_tokens)
        recall = lcs_len / len(g_tokens) if g_tokens else 0
        precision = lcs_len / len(p_tokens) if p_tokens else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0
        scores.append(f1)
    return {"rougeL": float(np.mean(scores)) if scores else 0.0}


def compute_meteor(preds: List[str], golds: List[str]) -> float:
    scores = []
    for pred, gold in zip(preds, golds):
        try:
            score = meteor_score([normalize_text(gold).split()], normalize_text(pred).split())
        except Exception:
            score = 0.0
        scores.append(score)
    return float(np.mean(scores)) if scores else 0.0

def compute_metrics(preds: List[str], golds: List[str]) -> Dict[str, float]:
    results = {}

    # Accuracy
    results["accuracy"] = compute_accuracy(preds, golds)

    # Precision, Recall, F1
    results.update(compute_precision_recall_f1(preds, golds))

    # BLEU
    results["bleu"] = compute_bleu(preds, golds)

    # ROUGE-L
    results.update(compute_rouge(preds, golds))

    # METEOR
    results["meteor"] = compute_meteor(preds, golds)

    return results


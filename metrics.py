"""Error-rate metrics implemented from scratch (no jiwer): WER, CER, alignment."""
import re


def normalize(text):
    text = text.lower()
    text = re.sub(r"[^\w\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def align(ref, hyp):
    """Levenshtein alignment on token lists.

    Returns (ops, counts) where ops is a list of (tag, ref_tok, hyp_tok) with
    tag in {'C','S','D','I'} and counts is a dict of C/S/D/I totals.
    """
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1):
            tag = "C" if ref[i - 1] == hyp[j - 1] else "S"
            ops.append((tag, ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(("D", ref[i - 1], ""))
            i -= 1
        else:
            ops.append(("I", "", hyp[j - 1]))
            j -= 1
    ops.reverse()

    counts = {"C": 0, "S": 0, "D": 0, "I": 0}
    for tag, _, _ in ops:
        counts[tag] += 1
    return ops, counts


def wer(reference, hypothesis):
    """Word error rate = (S + D + I) / N. Returns dict with rate, counts, ops."""
    ref = normalize(reference).split()
    hyp = normalize(hypothesis).split()
    ops, counts = align(ref, hyp)
    n = max(len(ref), 1)
    rate = (counts["S"] + counts["D"] + counts["I"]) / n
    return {"wer": rate, "counts": counts, "ref_words": len(ref), "ops": ops}


def cer(reference, hypothesis):
    """Character error rate over normalized text (spaces kept)."""
    ref = list(normalize(reference))
    hyp = list(normalize(hypothesis))
    ops, counts = align(ref, hyp)
    n = max(len(ref), 1)
    return {"cer": (counts["S"] + counts["D"] + counts["I"]) / n, "counts": counts}


def render_alignment(ops):
    """Return a list of (word, tag) pairs for colored display."""
    out = []
    for tag, r, h in ops:
        if tag == "C":
            out.append((r, "C"))
        elif tag == "S":
            out.append((f"{r}→{h}", "S"))
        elif tag == "D":
            out.append((r, "D"))
        else:
            out.append((h, "I"))
    return out

"""Sanity tests for metrics.py. Run:  python test_metrics.py"""
import metrics

r = metrics.wer("the cat sat on the mat", "the cat sat on the mat")
assert r["wer"] == 0.0 and r["counts"] == {"C": 6, "S": 0, "D": 0, "I": 0}

r = metrics.wer("the cat sat on the mat", "the cat sit on mat")
assert r["counts"]["S"] == 1 and r["counts"]["D"] == 1 and r["counts"]["I"] == 0
assert abs(r["wer"] - 2 / 6) < 1e-9

r = metrics.wer("hello world", "hello big wide world")
assert r["counts"]["I"] == 2 and abs(r["wer"] - 1.0) < 1e-9

r = metrics.wer("Hello, World!", "hello world")
assert r["wer"] == 0.0

r = metrics.cer("kitten", "sitting")
assert r["counts"]["S"] == 2 and r["counts"]["I"] == 1

assert metrics.wer("", "")["wer"] == 0.0
print("all metric tests passed")

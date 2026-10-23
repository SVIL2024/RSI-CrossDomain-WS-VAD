"""Recompute the five-seed paired statistics that go into the manuscript.

Reads nothing; the per-seed pooled AUC1 values are pasted in from
runs/cross_eval_xd2ucf_*.json on the experiment server (see dump_eval_jsons.py).
Prints mean +- s.d., paired differences, exact one-sided sign test and a
percentile bootstrap CI of the mean paired difference.
"""
import itertools
import random
from math import comb

SEEDS = [234, 555, 789, 222, 999]

ARMS = {
    "E5": [0.7965, 0.7901, 0.7812, 0.7674, 0.7739],
    "E3": [0.8202, 0.7995, 0.8007, 0.7693, 0.7895],
    "E1": [0.7980, 0.7923, 0.7775, 0.7683, 0.7838],
    "N1": [0.7926, 0.7842, 0.7887, 0.7469, 0.7508],
}

RTFM = {
    "E5": [0.7995, 0.7276, 0.8097],
    "E3": [0.6820, 0.6450, 0.7715],
}
RTFM_AP = {"E5": [0.7829, 0.4389, 0.7908], "E3": [0.5009, 0.3136, 0.6902]}


def mean(xs):
    return sum(xs) / len(xs)


def sd(xs):
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def sign_test(diffs):
    pos = sum(1 for d in diffs if d > 0)
    n = len(diffs)
    # one-sided p for "more positive than chance"
    p = sum(comb(n, k) for k in range(pos, n + 1)) / 2 ** n
    return pos, n, p


def boot_ci(diffs, n_boot=20000, alpha=0.05, seed=0):
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(mean([rng.choice(diffs) for _ in range(n)]) for _ in range(n_boot))
    lo = means[int(alpha / 2 * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot) - 1]
    return lo, hi


def report(name, xs, base=None):
    print("%-4s mean=%.4f sd=%.4f" % (name, mean(xs), sd(xs)), end="")
    if base is not None:
        d = [(a - b) * 100 for a, b in zip(xs, base)]
        lo, hi = boot_ci(d)
        pos, n, p = sign_test(d)
        print("  delta=%+.2f+-%.2f  diffs=%s  sign %d/%d p=%.4f  CI[%+.2f,%+.2f]"
              % (mean(d), sd(d), " ".join("%+.2f" % v for v in d), pos, n, p, lo, hi))
    else:
        print()


if __name__ == "__main__":
    base = ARMS["E5"]
    for k in ("E5", "E3", "E1", "N1"):
        report(k, ARMS[k], None if k == "E5" else base)

    print("\nE3 vs E1 (paired):")
    report("E3-E1", ARMS["E3"], ARMS["E1"])

    print("\nthree-seed subset (234/555/789):")
    sub = [0, 1, 2]
    for k in ("E5", "E3", "E1", "N1"):
        xs = [ARMS[k][i] for i in sub]
        report(k, xs, None if k == "E5" else [base[i] for i in sub])

    print("\nRTFM (three seeds, 30 epochs):")
    d = [(a - b) * 100 for a, b in zip(RTFM["E3"], RTFM["E5"])]
    print("  baseline mean=%.4f  E3 mean=%.4f  delta=%+.2f (per seed %s)"
          % (mean(RTFM["E5"]), mean(RTFM["E3"]), mean(d), " ".join("%+.2f" % v for v in d)))
    for k in ("E5", "E3"):
        print("  %s AP1 mean=%.4f  %s" % (k, mean(RTFM_AP[k]),
                                          " ".join("%.4f" % v for v in RTFM_AP[k])))

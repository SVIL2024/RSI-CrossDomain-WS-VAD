"""Recompute the paired statistics that go into the manuscript.

Reads nothing; the per-seed pooled AUC1 values are pasted in from
runs/cross_eval_xd2ucf_*.json on the experiment server (see dump_eval_jsons.py),
and the in-domain AP figures are the five-run means of logs/xd_*.log as printed
by final_table.py.  Prints mean +- s.d., paired differences, an exact one-sided
sign test and a percentile bootstrap CI of the mean paired difference.

Usage:  python five_seed_stats.py [--n-boot 200000]
"""
import argparse
import random
from math import comb

SEEDS = [222, 234, 555, 789, 999]

# pooled AUC1 on XD->UCF, class-agnostic head, one entry per seed above
ARMS = {
    "E5":  [0.7674, 0.7965, 0.7901, 0.7812, 0.7739],
    "E1":  [0.7683, 0.7980, 0.7923, 0.7775, 0.7838],
    "E3":  [0.7693, 0.8202, 0.7995, 0.8007, 0.7895],
    "N1":  [0.7469, 0.7926, 0.7842, 0.7887, 0.7508],
    "E3c": [0.7645, 0.7963, 0.8001, 0.7825, 0.7859],
    "E3w": [0.7618, 0.8041, 0.8012, 0.7865, 0.7934],
    "N2":  [0.7560, 0.8110, 0.7991, 0.8001, 0.7394],
}

# in-domain XD AP, five-run means (final_table.py over logs/xd_*.log)
AP_MEAN = {"E5": 0.8288, "E1": 0.8036, "E3": 0.8115, "N1": 0.7969,
           "E3c": 0.8151, "E3w": 0.8170, "N2": 0.8117}

# three-seed gate ablations (seeds 234/555/789, indices 1..3 of SEEDS)
GATE = {
    "E3a": [0.8012, 0.8417, 0.7965],
    "E3b": [0.8189, 0.7993, 0.7904],
}
GATE_BASE = [0.7965, 0.7901, 0.7812]
GATE_E3 = [0.8202, 0.7995, 0.8007]

RTFM = {
    "E5": [0.7995, 0.7276, 0.8097],
    "E3": [0.6820, 0.6450, 0.7715],
}
RTFM_AP = {"E5": [0.7829, 0.4389, 0.7908], "E3": [0.5009, 0.3136, 0.6902]}

N_BOOT = 200000


def mean(xs):
    return sum(xs) / len(xs)


def sd(xs):
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def sign_test(diffs):
    pos = sum(1 for d in diffs if d > 0)
    n = len(diffs)
    p = sum(comb(n, k) for k in range(pos, n + 1)) / 2 ** n
    return pos, n, p


def boot_ci(diffs, alpha=0.05, seed=0):
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(mean([rng.choice(diffs) for _ in range(n)])
                   for _ in range(N_BOOT))
    lo = means[int(alpha / 2 * N_BOOT)]
    hi = means[int((1 - alpha / 2) * N_BOOT) - 1]
    return lo, hi


def report(name, xs, base=None):
    print("%-6s mean=%.4f sd=%.4f" % (name, mean(xs), sd(xs)), end="")
    if base is not None:
        d = [(a - b) * 100 for a, b in zip(xs, base)]
        lo, hi = boot_ci(d)
        pos, n, p = sign_test(d)
        print("  delta=%+.2f+-%.2f  diffs=%s  sign %d/%d p=%.4f  CI[%+.2f,%+.2f]"
              % (mean(d), sd(d), " ".join("%+.2f" % v for v in d),
                 pos, n, p, lo, hi))
    else:
        print()


def main():
    global N_BOOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    N_BOOT = ap.parse_args().n_boot

    base = ARMS["E5"]
    print("== five seeds, pooled AUC1 (XD->UCF) ==")
    for k in ("E5", "E1", "E3", "N1", "E3c", "E3w", "N2"):
        report(k, ARMS[k], None if k == "E5" else base)

    print("\n== in-domain XD AP (five-run means) ==")
    for k in ("E5", "E1", "E3", "N1", "E3c", "E3w", "N2"):
        print("  %-4s %.4f" % (k, AP_MEAN[k]))

    print("\n== E3 vs E1 (paired, five seeds) ==")
    report("E3-E1", ARMS["E3"], ARMS["E1"])

    print("\n== decomposition against the full method (paired, five seeds) ==")
    for k, label in (("E3c", "frame selection"),
                     ("N2", "rank-1 direction"),
                     ("E3w", "contiguous window")):
        report("E3-%s" % k, ARMS["E3"], ARMS[k])

    print("\n== three-seed subset 234/555/789 ==")
    sub = [1, 2, 3]
    for k in ("E5", "E3"):
        report(k, [ARMS[k][i] for i in sub],
               None if k == "E5" else [base[i] for i in sub])

    print("\n== gate ablations (three seeds) ==")
    report("E3a", GATE["E3a"], GATE_BASE)
    report("E3b", GATE["E3b"], GATE_BASE)
    report("E3", GATE_E3, GATE_BASE)
    report("E3-E3b", GATE_E3, GATE["E3b"])
    report("E3a-E3b", GATE["E3a"], GATE["E3b"])

    print("\n== RTFM, 30 epochs (three seeds) ==")
    d = [(a - b) * 100 for a, b in zip(RTFM["E3"], RTFM["E5"])]
    print("  baseline mean=%.4f  E3 mean=%.4f  delta=%+.2f (per seed %s)"
          % (mean(RTFM["E5"]), mean(RTFM["E3"]), mean(d),
             " ".join("%+.2f" % v for v in d)))
    for k in ("E5", "E3"):
        print("  %s AP1 mean=%.4f  %s" % (k, mean(RTFM_AP[k]),
                                          " ".join("%.4f" % v for v in RTFM_AP[k])))


if __name__ == "__main__":
    main()

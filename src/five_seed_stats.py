"""Recompute the paired statistics that go into the manuscript.

Per-seed pooled AUC1 values (XD -> UCF, class-agnostic head) are pasted from
runs/cross_eval_xd2ucf_*.json on the experiment server, one entry per
(variant, seed).  Arms are compared on the seeds they share, which is what the
manuscript table captions promise: six runs for the headline decomposition
(E5, E3, N2, including the pre-registered replication at seed 777), five for the
rest of the attribution, three for the gate ablations.

Prints mean +- s.d., paired differences, an exact one-sided sign test and a
percentile bootstrap CI of the mean paired difference.  At n=5 there are only
3125 distinct resamples, so n_boot well below that is pure sampling noise; the
default of 200000 is what the published intervals use.

Usage:  python five_seed_stats.py [--n-boot 200000]
"""
import argparse
import random
from math import comb

# variant -> {seed: pooled AUC1 on XD -> UCF}
ARMS = {
    "E5":  {234: 0.7965, 555: 0.7901, 789: 0.7812, 222: 0.7674, 999: 0.7739, 777: 0.7507},
    "E3":  {234: 0.8202, 555: 0.7995, 789: 0.8007, 222: 0.7693, 999: 0.7895, 777: 0.7774},
    "N2":  {234: 0.8110, 555: 0.7991, 789: 0.8001, 222: 0.7560, 999: 0.7394, 777: 0.7965},
    "E1":  {234: 0.7980, 555: 0.7923, 789: 0.7775, 222: 0.7683, 999: 0.7838},
    "N1":  {234: 0.7926, 555: 0.7842, 789: 0.7887, 222: 0.7469, 999: 0.7508},
    "E3c": {234: 0.7963, 555: 0.8001, 789: 0.7825, 222: 0.7645, 999: 0.7859},
    "E3w": {234: 0.8041, 555: 0.8012, 789: 0.7865, 222: 0.7618, 999: 0.7934},
    "E3a": {234: 0.8012, 555: 0.8417, 789: 0.7965},
    "E3b": {234: 0.8189, 555: 0.7993, 789: 0.7904},
}

# in-domain XD AP, mean over the same runs as the AUC entry above
AP_MEAN = {
    "E5": 0.8257, "E3": 0.8147, "N2": 0.8101, "E1": 0.8036, "N1": 0.7969,
    "E3c": 0.8151, "E3w": 0.8170,
}

# (candidate, reference) contrasts reported in the manuscript
CONTRASTS = [
    ("E3", "E5"),   # headline, six runs
    ("E3", "N2"),   # is rank-1 sharing load-bearing? six runs
    ("N2", "E5"),
    ("E3", "E1"),   # semantics do not transfer, five runs
    ("N1", "E5"),   # energy alone does nothing, five runs
    ("E3", "E3c"),  # frame-selection rule, five runs
    ("E3", "E3w"),  # selection geometry, five runs
    ("E3", "E3a"),  # gate shape, three runs
    ("E3", "E3b"),  # gate learnability, three runs
]

RTFM = {"E5": [0.7995, 0.7276, 0.8097], "E3": [0.6820, 0.6450, 0.7715]}
RTFM_AP = {"E5": [0.7829, 0.4389, 0.7908], "E3": [0.5009, 0.3136, 0.6902]}


def mean(xs):
    return sum(xs) / len(xs)


def sd(xs):
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def sign_test(diffs):
    """One-sided exact binomial p for 'more positive than chance'."""
    pos = sum(1 for d in diffs if d > 0)
    n = len(diffs)
    return pos, n, sum(comb(n, k) for k in range(pos, n + 1)) / 2.0 ** n


def boot_ci(diffs, n_boot, alpha=0.05, seed=0):
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(mean([rng.choice(diffs) for _ in range(n)]) for _ in range(n_boot))
    return means[int(alpha / 2 * n_boot)], means[int((1 - alpha / 2) * n_boot) - 1]


def shared(a, b):
    return sorted(set(ARMS[a]) & set(ARMS[b]))


def report_arm(name):
    seeds = sorted(ARMS[name])
    xs = [ARMS[name][s] for s in seeds]
    ap = AP_MEAN.get(name)
    print("%-4s n=%d mean=%.4f sd=%.4f%s  seeds=%s"
          % (name, len(xs), mean(xs), sd(xs),
             "" if ap is None else "  AP=%.4f" % ap,
             " ".join(str(s) for s in seeds)))


def report_contrast(a, b, n_boot):
    seeds = shared(a, b)
    d = [(ARMS[a][s] - ARMS[b][s]) * 100 for s in seeds]
    lo, hi = boot_ci(d, n_boot)
    pos, n, p = sign_test(d)
    print("%-8s n=%d delta=%+.2f+-%.2f  diffs=%s  sign %d/%d p=%.4f  CI[%+.2f,%+.2f]"
          % (a + "-" + b, n, mean(d), sd(d),
             " ".join("%+.2f" % v for v in d), pos, n, p, lo, hi))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=200000)
    args = ap.parse_args()

    print("arms (pooled AUC1, XD->UCF; AP = in-domain XD AP over the same runs)")
    for v in ("E5", "E1", "E3", "N1", "E3c", "E3w", "N2", "E3a", "E3b"):
        report_arm(v)

    print("\npaired contrasts (on the seeds each pair shares)")
    for a, b in CONTRASTS:
        report_contrast(a, b, args.n_boot)

    print("\nthree-seed subset 234/555/789 (the original sweep)")
    for v in ("E5", "E3", "N2"):
        d = [(ARMS[v][s] - ARMS["E5"][s]) * 100 for s in (234, 555, 789)]
        print("  %-3s delta=%+.2f+-%.2f" % (v, mean(d), sd(d)))

    print("\nRTFM, second backbone (30 epochs, seeds 234/555/789)")
    print("  baseline mean=%.4f sd=%.4f  E3 mean=%.4f sd=%.4f"
          % (mean(RTFM["E5"]), sd(RTFM["E5"]), mean(RTFM["E3"]), sd(RTFM["E3"])))
    d = [(a - b) * 100 for a, b in zip(RTFM["E3"], RTFM["E5"])]
    print("  delta=%+.2f+-%.2f  per seed %s"
          % (mean(d), sd(d), " ".join("%+.2f" % v for v in d)))
    for k in ("E5", "E3"):
        print("  %s AP mean=%.4f  per seed %s"
              % (k, mean(RTFM_AP[k]), " ".join("%.4f" % v for v in RTFM_AP[k])))

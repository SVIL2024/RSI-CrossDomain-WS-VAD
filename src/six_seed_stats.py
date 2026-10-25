"""Recompute every statistic the manuscript reports on XD->UCF.

Seven arms, six seeds (222/234/555/777/789/999).  Values are the pooled AUC1
figures copied from runs/cross_eval_xd2ucf_*.json on the experiment server; the
seed-777 row was added last and E3 at seed 789 is the corrected 0.7953, the value
its stored checkpoint reproduces on re-evaluation (three runs: 0.7950/0.7953/
0.7953); the 0.8007 in the earlier run sheet could not be reproduced.

For every arm: mean, s.d., paired difference against the baseline, one-sided
exact sign test, and a percentile bootstrap CI of the mean paired difference.
Then the four contrasts the attribution argues over, and the correlation between
baseline strength and gain that separates the variants.

    python six_seed_stats.py [--n-boot 200000]
"""
import argparse
import random
from math import comb

SEEDS = [234, 555, 789, 222, 999, 777]

AUC1 = {
    "E5": [0.7965, 0.7901, 0.7812, 0.7674, 0.7739, 0.7507],
    "E3": [0.8202, 0.7995, 0.7953, 0.7693, 0.7895, 0.7774],
    "N2": [0.8110, 0.7991, 0.8001, 0.7560, 0.7394, 0.7965],
    "E1": [0.7980, 0.7923, 0.7775, 0.7683, 0.7838, 0.7813],
    "N1": [0.7926, 0.7842, 0.7887, 0.7469, 0.7508, 0.7846],
    "E3c": [0.7963, 0.8001, 0.7825, 0.7645, 0.7859, 0.7782],
    "E3w": [0.8041, 0.8012, 0.7865, 0.7618, 0.7934, 0.7829],
}

# In-domain XD AP1 of the selected checkpoint, read from its filename.
AP1 = {
    "E5": [0.8456, 0.8377, 0.8308, 0.8315, 0.8178, 0.7909],
    "E3": [0.8413, 0.8164, 0.8102, 0.8006, 0.8186, 0.8012],
    "N2": [0.8422, 0.8073, 0.7981, 0.7771, 0.8336, 0.8022],
    "E1": [0.8410, 0.8008, 0.8065, 0.7912, 0.8161, 0.8102],
    "N1": [0.8312, 0.8051, 0.7985, 0.7634, 0.8288, 0.7952],
    "E3c": [0.8440, 0.8162, 0.8001, 0.7870, 0.8284, 0.8023],
    "E3w": [0.8482, 0.8207, 0.8169, 0.7929, 0.8061, 0.8021],
}

CONTRASTS = [("E3", "E1"), ("E3", "N2"), ("E3", "E3c"), ("E3", "E3w")]


def mean(xs):
    return sum(xs) / len(xs)


def sd(xs):
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def sign_test(diffs):
    pos = sum(1 for d in diffs if d > 0)
    n = len(diffs)
    return pos, n, sum(comb(n, k) for k in range(pos, n + 1)) / 2 ** n


def boot_ci(diffs, n_boot, alpha=0.05, seed=0):
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(mean([rng.choice(diffs) for _ in range(n)]) for _ in range(n_boot))
    return means[int(alpha / 2 * n_boot)], means[int((1 - alpha / 2) * n_boot) - 1]


def corr(x, y):
    mx, my = mean(x), mean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den = (sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y)) ** 0.5
    return num / den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=200000)
    args = ap.parse_args()

    base = AUC1["E5"]
    print("XD->UCF pooled AUC1, six seeds %s\n" % SEEDS)
    print("%-4s %-16s %-9s %-6s %-8s %-20s %s"
          % ("arm", "mean sd", "d(E5)", "pos/n", "p", "95% CI", "XD AP"))
    for k in ("E5", "N1", "E3c", "E3w", "N2", "E1", "E3"):
        xs = AUC1[k]
        if k == "E5":
            print("%-4s %.4f +- %.4f  %-9s %-6s %-8s %-20s %.4f"
                  % (k, mean(xs), sd(xs), "--", "--", "--", "--", mean(AP1[k])))
            continue
        d = [(a - b) * 100 for a, b in zip(xs, base)]
        pos, n, p = sign_test(d)
        lo, hi = boot_ci(d, args.n_boot)
        print("%-4s %.4f +- %.4f  %+7.2f   %d/%d   %.4f   [%+.2f,%+.2f]      %.4f"
              % (k, mean(xs), sd(xs), mean(d), pos, n, p, lo, hi, mean(AP1[k])))

    print("\nPaired contrasts (per-run differences, in AUC points):")
    for a, b in CONTRASTS:
        d = [(x - y) * 100 for x, y in zip(AUC1[a], AUC1[b])]
        pos, n, p = sign_test(d)
        lo, hi = boot_ci(d, args.n_boot)
        print("  %s-%s  %+.2f +- %.2f   %s   %d/%d  p=%.4f  CI[%+.2f,%+.2f]"
              % (a, b, mean(d), sd(d), " ".join("%+.2f" % v for v in d),
                 pos, n, p, lo, hi))

    print("\nCorrelation between baseline AUC and gain (per seed):")
    for k in ("E1", "E3c", "E3w", "N1", "N2", "E3"):
        d = [(AUC1[k][i] - base[i]) * 100 for i in range(len(base))]
        print("  %-4s %+.2f" % (k, corr(base, d)))


if __name__ == "__main__":
    main()
